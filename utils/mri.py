import torch
import utils.mri
from medutils.visualization import center_crop, imshow, plot_array
import matplotlib.pyplot as plt
from skimage.transform import rescale
import numpy as np
import torchkbnufft as tkbn
from utils.basic import torch2numpy, numpy2torch
from tqdm import tqdm

def ifft2c_mri(k):
    x = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k, (-2,-1)), norm='ortho'), (-2,-1))
    return x

def fft2c_mri(img):
    k = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img, (-2,-1)), norm='ortho'), (-2,-1))
    return k

def coilcombine(img, im_shape=None, coil_dim=-1, mode='csm', csm=None):
    if mode == 'rss':
        return torch.sqrt(torch.sum(img**2, dim=coil_dim, keepdim=True))
    elif mode == 'csm':
        # csm = csm.unsqueeze(0)
        csm = torch.from_numpy(csm).to(img.device)
        img = center_crop(img, im_shape)
        csm = center_crop(csm, im_shape)
        return torch.sum(img*torch.conj(csm), dim=coil_dim, keepdim=True)
    else:
        raise NotImplementedError

def downsample_csm(csm, factor):
    # Downsample images
    # Separate the real and imaginary parts
    csm_real = np.real(csm)
    csm_imag = np.imag(csm)
    # Downsample the real and imaginary parts
    csm_real_downsampled = rescale(csm_real, (factor, factor), mode='reflect', channel_axis=0)
    csm_imag_downsampled = rescale(csm_imag, (factor, factor), mode='reflect', channel_axis=0)
    # Combine the downsampled real and imaginary parts back into a complex array
    csm_downsampled = csm_real_downsampled + 1j * csm_imag_downsampled

    imshow(plot_array(csm), title= "Original")
    imshow(plot_array(csm_downsampled), title= "Downsampled")
    plt.show()

    return csm_downsampled

def scale_traj(traj, max_value=np.pi):
    """Scale an array to the range [-1, 1] based on its maximum absolute value."""
    if isinstance(traj, np.ndarray):
        x = np.max(np.abs(traj))
    elif isinstance(traj, torch.Tensor):
        x = torch.max(torch.abs(traj))

    # To prevent division by zero in case the array is full of zeros
    if x == 0:
        return traj

    scaling_factor = max_value / x
    new_traj = scaling_factor * traj
    return new_traj

def software_coil_compression(kdata, ncc=None):
    _, S, Vh = torch.linalg.svd(kdata, full_matrices=False)
    cc_mat = Vh.mH[:, :ncc] # matrix hermitian back
    kdata_cc = kdata @ cc_mat
    explained_variance_ratio = torch.sum(S[:ncc]) / torch.sum(S)
    return kdata_cc, cc_mat, explained_variance_ratio

def coil_compression_torch(kspace, ncc=None, s_ech=0):
    '''
    input:
        kspace: torch.tensor or np.ndarray [nEch, nCoils, *img_dim]
        ncc: number of coils to be compressed tp
        s_ech: selected echo used for compression
    returns:
    - kspace: torch.tensor [nEch, ncoils, *img_dim]
    '''
    if isinstance(kspace, np.ndarray):
        kspace = torch.from_numpy(kspace)

    nEch, nCoils, dim1, dim2, dim3 = kspace.shape
    # orig_size = kspace.shape
    kspace = kspace.permute(0,2,3,4,1)          # move coils to last dim
    kspace = kspace.reshape(nEch, -1, nCoils) # flatten all sample points

    kspaceCC = torch.zeros_like(kspace[:,:,:ncc])
    kspaceCC[s_ech,...], ccMat, exp_ratio = software_coil_compression(kspace[s_ech, ...], ncc)

    for ech in range(nEch):
        if ech == s_ech:
            continue
        else:
            kspaceCC[ech, ...] = kspace[s_ech, ...] @ ccMat

    kspaceCC = kspaceCC.reshape(nEch, dim1, dim2, dim3, ncc)
    kspaceCC = kspaceCC.permute(0,4,1,2,3)
    return kspaceCC.to("cpu"), ccMat, exp_ratio


def coil_compression_slicewise_torch(kspace, ncc=None, min_explained_ratio=None, device="cuda:0", chunk_size=10, s_ech=0):
    '''
    input:
        kspace: torch.tensor or np.ndarray [nEch, nCoils, nSlices, *img_dim]
        ncc: number of coils to be compressed to, if None: all with less than 0.05 * S[0] will be discarded
        coil_dim: coil dimension, all extra dims before will treated individually, all after are used for compression
    returns:
    - kspace: torch.tensor [ncoils, *img_dim]
    - explained_variance_ration: np.float - how much information of original kdata is kept in compressed version
    '''

    if isinstance(kspace, np.ndarray):
        kspace = torch.from_numpy(kspace)

    nEch, nCoils, nSlices, nPE, nFE = kspace.shape
    kdataCC = torch.zeros_like(kspace[:,:ncc,...]) # nEch, nCoils, nSlices, nPE, nFE
    expRatio = torch.zeros(nSlices,1)

    for sl in range(nSlices):
        kdataCC[:,:,[sl],:,:], ccMat, expRatio[sl] = coil_compression_torch(kspace[:,:,[sl],:,:].to(device), ncc=ncc, s_ech=s_ech)

    return kdataCC, expRatio

def geometric_coil_compression_torch(kspace, ncc=None, coil_dim=1, s_ech=0, readout_dim=-1):

    '''
    input:
        kspace: torch.tensor or np.ndarray [nEch, nCoils, nSlices] with img_dim = nPE, nFE or flipped
        ncc: number of coils to be compressed to, if None: all with less than 0.05 * S[0] will be discarded
        coil_dim: coil dimension, all extra dims before will treated individually, all after are used for compression
        readout_dim: readout dimension which is used to perform FFT and coil compression
    returns:
    - kspace: torch.tensor [ncoils, *img_dim]
    - explained_variance_ration: np.float - how much information of original kdata is kept in compressed version
    '''

    if isinstance(kspace, np.ndarray):
        kspace = torch.from_numpy(kspace)

    if readout_dim == -1:
        pass
    elif readout_dim == -2:
        kspace = kspace.permute(0,1,2,4,3)

    nEch, nCoils, dim1, dim2, dim_ro = kspace.shape
    kspace = kspace.permute(4,3,2,1,0) # bring readout to front and channel to back (dim_ro, dim2, dim1, nCoils, nEch)


    # perform ifft along last dimension
    calibData = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace, axis=0),axis=0),axis=0)
    calibData = calibData.view(dim_ro, dim2*dim1, nCoils, nEch)

    kspaceCC = torch.zeros_like(calibData[..., :ncc,:])
    ccMat = torch.zeros((dim_ro, nCoils, ncc), dtype=kspace.dtype).to(kspace)

    for kx in range(dim_ro):
        _,S,Vh = torch.linalg.svd(calibData[kx, :, :, s_ech], full_matrices=False)
        ccMat[kx,:,:] = Vh.mH[:,:ncc]

    # align ccMat
    ccMat = align_cc_mtx(ccMat, num_vc=ncc)
    # apply ccMat
    for kx in range(dim_ro):
        for ech in range(nEch):
            kspaceCC[kx,:,:,ech] = calibData[kx, ..., ech] @ ccMat[kx,:,:]

    kspaceCC = kspaceCC.view(dim_ro, dim2, dim1, ncc, nEch)
    kspaceCC = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(kspaceCC, dim = 0),dim=0), dim=0)

    # permute data back if compression was done on the 2nd or 3rd dimensions
    kspaceCC = kspaceCC.permute(4,3,2,1,0) # nEch, nCoils, dim1, dim2, rho

    if readout_dim == -1:
        pass
    elif readout_dim == -2:
        kspaceCC = kspace.permute(0,1,2,4,3)

    return kspaceCC

def geometric_coil_compression_slicewise_torch(kspace, ncc=None, coil_dim=1, s_ech=0, readout_dim=-1):
    if isinstance(kspace, np.ndarray):
        kspace = torch.from_numpy(kspace)
    nEch, nCoils, nSlices, nPE, nFE = kspace.shape
    kdataCC = torch.zeros_like(kspace[:,:ncc,...]) # nEch, nCoils, nSlices, nPE, nFE
    exp_ratio = torch.zeros(nSlices)
    for sl in range(nSlices):
        kdataCC[:,:,[sl],:,:] = geometric_coil_compression_torch(kspace[:,:,[sl],:,:], ncc=ncc, s_ech=s_ech)
    return kdataCC, exp_ratio # ToDO: explainability ratio

def align_cc_mtx(cc_mat, num_vc=None):
    sx, _, _ = cc_mat.shape

    # align everything based on the middle slice.
    n0 = np.floor(sx / 2).astype(np.int32)
    A00 = cc_mat[n0, :, :num_vc]

    # Align backward to first slice
    A0 = A00.clone()
    for n in range(n0-1, -1, -1):
        A1 = cc_mat[n, :, :num_vc]
        C = A1.T @ A0
        U, S, Vh = torch.linalg.svd(C)
        P = Vh.mH @ U.T   # V * U'
        cc_mat[n, :, :num_vc] = A1 @ P
        A0 = cc_mat[n, :, :num_vc]

    # Align forward to end slice
    A0 = A00.clone()
    for n in range(n0+1, sx):
        A1 = cc_mat[n, :, :num_vc]
        C = A1.T @ A0
        U, S, Vh = torch.linalg.svd(C)
        P = Vh.mH @ U.T  # V * U'
        cc_mat[n, :, :num_vc] = A1 @ P
        A0 = cc_mat[n, :, :num_vc]

    return cc_mat

def mriRadialForwardOp(img, shape, traj, dcf=None, csm=None, coil_axis=1, norm = True, osf = 2, device="cuda:0"):

    # img, traj, csm = numpy2torch(img, device), numpy2torch(traj, device), numpy2torch(csm, device)
    img, traj = numpy2torch(img, device), numpy2torch(traj, device)
    traj = traj.to(img.real.dtype)
    assert torch.is_complex(img) and img.real.dtype == traj.dtype, "traj must have same precision as kspace data type"

    if csm is not None:
        csm = numpy2torch(csm, device)
        csm = csm.to(img.dtype)

    grid_size = [int(sz * osf) for sz in shape]
    nufft_op = tkbn.KbNufft(im_size=shape, grid_size=grid_size).to(img)

    if traj.max() is not torch.pi:
        print("rescaling trajectory for torchkbnufft from {} to pi".format(traj.max()))
        traj = utils.mri.scale_traj(traj, torch.pi)

    if type(dcf) == str and dcf == "calc":
        dcf = tkbn.calc_density_compensation_function(traj, im_size=shape)
    elif dcf is not None:
        dcf = numpy2torch(dcf,device)

    kspace = nufft_op(img, traj, smaps = csm)
    # kspace = nufft_op(img, traj, csm = None, norm = "ortho")
    kspace = kspace / torch.max(torch.abs(kspace)) if norm else kspace
    kspace = kspace * dcf if dcf is not None else kspace
    # kspace_joint = kspace_joint / kspace_joint.abs()
    return kspace

def mriRadialAdjointOp(kspace, shape, traj, dcf=None, csm = None, coil_axis = 1, osf = 2, chunk_size = None, device="cuda:0"):

    kspace, traj = numpy2torch(kspace, device), numpy2torch(traj, device)
    traj = traj.to(dtype=kspace.real.dtype)
    assert torch.is_complex(kspace) and kspace.real.dtype == traj.dtype, "traj must have same precision as kspace data type"

    if csm is not None:
        csm = numpy2torch(csm, device)
        csm = csm.to(kspace.dtype)

    grid_size = [int(sz * osf) for sz in shape]
    adj_nufft_op = tkbn.KbNufftAdjoint(im_size=shape, grid_size=grid_size).to(kspace)

    if traj.max() is not torch.pi:
        print("rescaling trajectory for torchkbnufft from {} to pi".format(traj.max()))
        traj = utils.mri.scale_traj(traj, torch.pi)

    if type(dcf) == str and dcf == "calc":
        dcf = tkbn.calc_density_compensation_function(traj, im_size=shape)
    elif dcf is not None:
        dcf = numpy2torch(dcf, device)
        dcf = dcf.to(dtype=kspace.dtype)

    kspace = kspace * dcf if dcf is not None else kspace

    if chunk_size is None:
        img = adj_nufft_op(kspace, traj, smaps = csm)
    else:
        nBatch = kspace.shape[0]
        img_list = []
        for start in tqdm(range(0, nBatch, chunk_size),
                                  desc='Processing batches of kspace slices to images'.format(chunk_size), leave=False):
            end = start + min(chunk_size, nBatch - start)
            csm_temp = csm[start:end,...] if csm is not None else csm
            img_list.append(adj_nufft_op(kspace[start:end, :], traj[start:end, :], smaps = csm_temp))
        img = torch.cat(img_list, dim=0)
    return img