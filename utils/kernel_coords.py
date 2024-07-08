import numpy as np
import torch

def cart2pol(coords):
    if torch.is_tensor(coords):
        return torch.stack([torch.sqrt(coords[..., 0] ** 2 + coords[..., 1] ** 2),  # Magnitude
                            torch.atan2(coords[..., 1], coords[..., 0])], dim=-1)  # Angle
    elif isinstance(coords, np.ndarray):
        return np.stack([np.sqrt(coords[..., 0] ** 2 + coords[..., 1] ** 2),  # Magnitude
                         np.arctan2(coords[..., 1], coords[..., 0])], axis=-1)  # Angle

def rotation_matrix(phi_array):
    cos_phi = np.cos(phi_array)
    sin_phi = np.sin(phi_array)
    rotation_matrix_elements = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
    return rotation_matrix_elements.transpose((2, 0, 1))

def pol2cart(coords):
    if torch.is_tensor(coords):
        return torch.stack([coords[..., 0] * torch.cos(coords[..., 1]),  # x-coordinate
                            coords[..., 0] * torch.sin(coords[..., 1])], dim=-1)  # y-coordinate
    elif isinstance(coords, np.ndarray):
        return np.stack([coords[..., 0] * np.cos(coords[..., 1]),  # x-coordinate
                         coords[..., 0] * np.sin(coords[..., 1])], axis=-1)  # y-coordinate


def get_edge_coords(coords, coords_patch, threshold):
    if torch.is_tensor(coords):
        coords = coords.cpu().numpy()
    if torch.is_tensor(coords_patch):
        coords_patch = coords_patch.cpu().numpy()
    assert coords.shape[0] == coords_patch.shape[0]
    if coords.ndim != 2:
        coords = coords.reshape(-1, coords.shape[-1])
    dist = np.sqrt(coords[...,1] ** 2 + coords[...,2] ** 2)
    dist_patch = np.sqrt(coords_patch[..., 1] ** 2 + coords_patch[..., 2] ** 2)
    edge_idx_target = dist > (1. - threshold)
    edge_idx_patch = np.any(dist_patch > (1. - threshold), axis=-1)
    edge_idx = np.logical_or(edge_idx_target, edge_idx_patch)
    # coords_filt = coords[valid_idx]
    return edge_idx


def get_center_coord(coords, coords_patch, threshold):
    if torch.is_tensor(coords):
        coords = coords.cpu().numpy()
    if torch.is_tensor(coords_patch):
        coords_patch = coords_patch.cpu().numpy()
    assert coords.shape[0] == coords_patch.shape[0]
    dist = np.sqrt(coords[..., 1] ** 2 + coords[..., 2] ** 2)
    dist_patch= np.sqrt(coords_patch[..., 1] ** 2 + coords_patch[..., 2] ** 2)
    center_idx_target = dist <= threshold
    center_idx_patch = np.any(dist_patch <= threshold, axis=-1)

    center_idx = np.logical_or(center_idx_target, center_idx_patch)
    # select = select == False  # flip select to only choose values without center
    return center_idx

def create_cartesian_kernel(coords, kernel_size, patch_dist, delta_dist, middle_index=False):

    if coords.ndim != 2:
        coords = coords.reshape(-1, coords.shape[-1])

    dpatch = [int(np.floor(k / 2)) for k in kernel_size]
    kernel = np.stack(np.meshgrid(np.linspace(-dpatch[0], dpatch[0], 2 * dpatch[0] + 1),
                                        np.linspace(-dpatch[1], dpatch[1], 2 * dpatch[1] + 1),
                                        indexing="ij"), axis=-1)

    # define spacing between neighboring points
    dx = patch_dist[0] * delta_dist[0]
    dy = patch_dist[1] * delta_dist[1]
    dt = 0  # patch_dist * 2.0 / self.config['fe_steps'] # ToDo: Adjust distance for t when adding t for kernel
    kernel[..., 0] = kernel[..., 0] * dx
    kernel[..., 1] = kernel[..., 1] * dy
    kernel_flat = np.reshape(kernel, (-1, 2))
    if middle_index:
        pass
    else:
        middle_index = len(kernel_flat) // 2
        kernel_flat = np.concatenate((kernel_flat[:middle_index], kernel_flat[middle_index + 1:]))

    coords_P = coords.copy()
    coord_neighbors = coords_P[:,None,...].repeat(kernel_flat.shape[0], axis=1)
    coord_neighbors[..., 1:] = coord_neighbors[..., 1:] + kernel_flat  ## adding kernel offset here
    return coord_neighbors

def create_cartesian_kernel_withoutput(coords, output, kernel_size, patch_dist):
    '''
    Creates kernels with neighors from given trajectory, i.e. coords and traj have the same size
    coords = nt, nx, ny, 3
    output = nx, ny, out_dim
    '''

    dpatch = [int(np.floor(k / 2)) for k in kernel_size]
    kernel = np.stack(np.meshgrid(np.linspace(-dpatch[0], dpatch[0], 2 * dpatch[0] + 1),
                                        np.linspace(-dpatch[1], dpatch[1], 2 * dpatch[1] + 1),
                                        indexing="ij"), axis=-1)
    kernel[...,0] = kernel[...,0] * patch_dist[0]
    kernel[...,1] = kernel[...,1] * patch_dist[1]

    kernel_flat = np.reshape(kernel, (-1, 2))
    middle_index = len(kernel_flat) // 2
    kernel_flat = np.concatenate((kernel_flat[:middle_index], kernel_flat[middle_index + 1:]))
    kernel_flat = kernel_flat.astype(int)

    coord_neighbors = np.zeros([*coords.shape[:-1], len(kernel_flat), coords.shape[-1]])
    output_neighbors = np.zeros([*output.shape[:-1], len(kernel_flat), output.shape[-1]], dtype=np.complex128)
    for k in range(len(kernel_flat)):
        coord_neighbors[...,k,:] = np.roll(coords, (-kernel_flat[k, 0],  -kernel_flat[k, 1]), axis=(1,2)) # careful: values bexond last position roll to first
        output_neighbors[...,k,:] = np.roll(output, (-kernel_flat[k, 0],  -kernel_flat[k, 1]), axis=(0,1))
    return coord_neighbors, output_neighbors


def create_radial_kernel(coords, kernel_size, patch_dist, delta_dist_rad, middle_index=False, half=False):

    dpatch = [int(np.floor(k / 2)) for k in kernel_size]

    # Frequency encoding points
    dfe_idx = np.linspace(-dpatch[0], dpatch[0], 2 * dpatch[0] + 1)
    # Phase encoding points
    if half:    # only half the kernel size (phase encoding only on one side of target considered)
        dfe_idx = np.linspace(-dpatch[0], 0, dpatch[0]) # middle index not considered immediately
    else:       # complete kernel (symmetric kernel)
        dphi_idx = np.linspace(-dpatch[0], dpatch[0], 2 * dpatch[0] + 1)
        if middle_index:
            pass
        else:
            middle_index = len(dphi_idx) // 2  # remove center spoke (where target lies on)
            dphi_idx = np.concatenate((dphi_idx[:middle_index], dphi_idx[middle_index + 1:]))


    kernel = np.stack(np.meshgrid(dfe_idx, dphi_idx, indexing="ij"), axis=-1)
    kernel[..., 0] = kernel[..., 0] * patch_dist[0] * delta_dist_rad[0] # delta fe
    kernel[..., 1] = kernel[..., 1] * patch_dist[1] * delta_dist_rad[1] # delta phi
    kernel_flat = np.reshape(kernel, (-1, 2))

    origin_coord = coords.copy()
    origin_coord_pol_xy = cart2pol(origin_coord[:, 1:])
    coord_neighbors_pol_xy = origin_coord_pol_xy[:, None, ...].repeat(kernel_flat.shape[0], axis=1)
    coord_neighbors_pol_xy[..., :] = coord_neighbors_pol_xy[...,:] + kernel_flat  ## adding fe [..., 0] and phi [..., 1] kernel offset here
    coord_neighbors = np.concatenate([origin_coord[:, None, [0]].repeat(kernel_flat.shape[0], axis=1),
                                      pol2cart(coord_neighbors_pol_xy)], axis=-1)

    return coord_neighbors

def create_radial_kernel_withoutput(coords, output, kernel_size, patch_dist):
    '''
    Creates kernels with neighors from given trajectory, i.e. coords and traj have the same size
    coords = nt, nx, ny, 3
    output = nx, ny, out_dim
    '''

    dpatch = [int(np.floor(k / 2)) for k in kernel_size]

    # Frequency encoding points
    dfe_idx = np.linspace(-dpatch[0], dpatch[0], 2 * dpatch[0] + 1)
    dphi_idx = np.linspace(-dpatch[1], dpatch[1], 2 * dpatch[1] + 1)
    middle_index = len(dphi_idx) // 2  # remove center spoke (where target lies on)
    dphi_idx = np.concatenate((dphi_idx[:middle_index], dphi_idx[middle_index + 1:]))

    kernel = np.stack(np.meshgrid(dfe_idx, dphi_idx, indexing="ij"), axis=-1)
    kernel[...,0] = kernel[...,0] * patch_dist[0]
    kernel[...,1] = kernel[...,1] * patch_dist[1]

    kernel_flat = np.reshape(kernel, (-1, 2))
    kernel_flat = kernel_flat.astype(int)

    nSpokes, nFE = coords.shape[1:3]
    delta_fe = 2.0 / nFE
    delta_phi = np.pi / nSpokes

    coords, output = coords[...].reshape(-1, coords.shape[-1]), output[...].reshape(-1, output.shape[-1])
    coords_neighbors = create_radial_kernel(coords, kernel_size=kernel_size, patch_dist=patch_dist,
                                            delta_dist_rad=[delta_fe,delta_phi])
    coords_neighbors = coords_neighbors.reshape(-1, 3)
    output_neighbors = find_nearest_neighbors_chunked(coords, coords_neighbors, output, chunk_size=1000)

    coords_neighbors = coords_neighbors.reshape(-1, len(kernel_flat), coords_neighbors.shape[-1])
    output_neighbors = output_neighbors.reshape(-1, len(kernel_flat), output_neighbors.shape[-1])

    return coords_neighbors, output_neighbors


def create_radial_equidistant_kernel(coords, kernel_size, patch_dist, delta_dist_rad, middle_index=False, half=False):

    # Create kernel
    dpatch = [int(np.floor(k / 2)) for k in kernel_size]
    dfe_idx = np.linspace(-dpatch[0], dpatch[0], 2 * dpatch[0] + 1)
    # Phase encoding points
    if half:    # only half the kernel size (phase encoding only on one side of target considered)
        dfe_idx = np.linspace(-dpatch[0], 0, dpatch[0]) # middle index not considered immediately
    else:       # complete kernel (symmetric kernel)
        dphi_idx = np.linspace(-dpatch[0], dpatch[0], 2 * dpatch[0] + 1)
        if middle_index:
            pass
        else:
            middle_index = len(dphi_idx) // 2  # remove center spoke (where target lies on)
            dphi_idx = np.concatenate((dphi_idx[:middle_index], dphi_idx[middle_index + 1:]))

    kernel = np.stack(np.meshgrid(dfe_idx, dphi_idx, indexing="ij"), axis=-1)
    kernel[..., 0] = kernel[..., 0] * patch_dist[0] * delta_dist_rad[0] # delta fe
    kernel[..., 1] = kernel[..., 1] * patch_dist[1] * delta_dist_rad[1] # delta phi
    kernel_flat = np.reshape(kernel, (-1, 2))

    ## calculate angle of each coord
    origin_coord = coords.copy().reshape(-1, coords.shape[-1])
    origin_coord_angle = cart2pol(origin_coord[:, 1:])[...,1] # extract angle
    origin_coord_rot = rotation_matrix(origin_coord_angle)
    coord_neighbors = origin_coord[:, None, ...].repeat(kernel_flat.shape[0], axis=1)
    coord_neighbors[...,1:] = coord_neighbors[...,1:] + np.matmul(kernel_flat, origin_coord_rot)

    return coord_neighbors
    ## add kernel multiplied by rotation matrix to original coord


def find_nearest_neighbors_chunked(coords, coords_neighbors, output, chunk_size=500, device="cuda:0"):
    ## Still quite memory intensive function but avoids dependency on the trajectory creation by indexing
    coords = torch.from_numpy(coords).to(device)
    coords_neighbors = torch.from_numpy(coords_neighbors).to(device)
    output = torch.from_numpy(output).to(device)

    num_chunks = (coords_neighbors.shape[0] + chunk_size - 1) // chunk_size
    output_neighbors = torch.empty((coords_neighbors.shape[0], output.shape[-1]), dtype=output.dtype)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, coords_neighbors.shape[0])

        chunk_indices = torch.argmin(torch.linalg.norm(coords - coords_neighbors[start_idx:end_idx, None, :], axis=2), axis=1)
        output_neighbors[start_idx:end_idx, :] = output[chunk_indices, :]

    return output_neighbors.cpu().numpy()