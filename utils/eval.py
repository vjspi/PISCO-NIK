import numpy as np
from medutils.measures import psnr, ssim, nrmse, nrmseAbs
import medutils

import matplotlib.pyplot as plt
import math
import wandb
import torch
from utils.vis import k2img, k2img_multiecho
from utils.basic import torch2numpy
import cv2
from image_similarity_measures.quality_metrics import fsim

def make_string_from_value_dict(eval_dict, default_keys = ["ssim", "psnr", "nrmse"]):
    ## values to string  # define values to extract
    values_text_list = []
    for key in default_keys:
        if key in eval_dict.keys():
            temp_val = eval_dict[key]
            values_text_list.append(f"{temp_val:.2f}")
        else:
            values_text_list.append("-")
        # values_text_list = "/".join(temp_list)
        # values_text_list.append(temp_text)
    values_text_list = "/".join(values_text_list)
    return values_text_list

def create_hystereses(img, dim_axis):
    img = torch2numpy(img)
    final_dim = 100
    orig_dim = img.shape[dim_axis]
    if orig_dim == 1:
        img_hyst = np.repeat(img, final_dim, axis=dim_axis)
    elif orig_dim == 4:
        recon_hyst = np.vstack((img, np.flip(img, axis=dim_axis)))  # 8 MS: 1,2,3,4,4,3,2,1
        repeat_counts = np.array([13, 12, 13, 12, 13, 12, 13, 12])
        img_hyst = np.concatenate([np.repeat(recon_hyst[i:i + 1], repeat_counts[i], axis=0)
                                                for i in range(len(repeat_counts))], axis=0)
    elif orig_dim == final_dim:
        img_hyst = img
    else:
        print("Hysteresis is applied for {} nMS".format(orig_dim))
        img_hyst = np.vstack((img, np.flip(img, axis=dim_axis)))

    return img_hyst

def postprocess(pred, ref = None):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    else:
        pred = pred
    assert pred.shape[1:] == ref.shape[1:]
    assert pred.shape[0] == ref.shape[0] or pred.shape[0] == ref.shape[0] // 2

    if ref is not None:
        ref = medutils.visualization.normalize(ref, max_int=1)

    pred = medutils.visualization.contrastStretching(pred)
    pred = np.stack([bias_corr(pred[i, ...], ref[i, ...], mag=False) for i in range(pred.shape[0])])  # loop over dynamics
    pred = medutils.visualization.normalize(pred, max_int=1)
    return pred

def postprocess_with_reference(pred, ref):

    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    else:
        pred = pred

    assert pred.shape[1:] == ref.shape[1:]
    assert pred.shape[0] == ref.shape[0] or pred.shape[0] == ref.shape[0] // 2

    # Normalize # ToDo: How ideally normalize?
    pred = medutils.visualization.contrastStretching(pred)
    pred = np.stack([bias_corr(pred[i, ...], ref[i, ...], mag=False) for i in range(pred.shape[0])])  # loop over dynamics
    if ref is not None:
        ref = medutils.visualization.normalize(ref, max_int=1)
        pred = medutils.visualization.normalize(pred, max_int=1)
        # ref_mag = np.abs(ref)
        # pred_corr_mag = np.abs(pred_corr)
        # ref_mag = (ref_mag - ref_mag.min()) / (ref_mag.max() - ref_mag.min())
        # pred_corr_mag = (pred_corr_mag - pred_corr_mag.min()) * ((ref_mag.max() - ref_mag.min()) / (pred_corr_mag - pred_corr_mag.min())) # rescale to reference

        pred = create_hysteresis(pred) if pred.shape[0] == ref.shape[0] // 2 else pred
        # pred = np.zeros_like(pred)


    return pred, ref

def postprocess_without_reference(pred):

    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    else:
        pred = pred

    # Normalize # ToDo: How ideally normalize?
    pred_corr = medutils.visualization.contrastStretching(pred)

    pred_corr_mag = np.abs(pred_corr)
    pred_corr_mag = (pred_corr_mag - pred_corr_mag.min()) / (pred_corr_mag.max() - pred_corr_mag.min()) # rescale to reference

    pred = create_hysteresis(pred) # if pred.shape[0] == ref.shape[0] // 2 else pred
    # pred = np.zeros_like(pred)
    # pred_corr = np.stack([bias_corr(pred[i, ...], ref[i, ...], mag=False)
    #                       for i in range(pred.shape[0])])  # loop over dynamics

    return pred


def get_rescaled_magnitude(pred_img, ref):

    if isinstance(pred_img, torch.Tensor):
        pred = pred_img.cpu().numpy()
    else:
        pred = pred_img

    assert pred.shape[1:] == ref.shape[1:]
    assert pred.shape[0] == ref.shape[0] or pred.shape[0] == ref.shape[0] // 2
    pred = create_hysteresis(pred) if pred.shape[0] == ref.shape[0] // 2 else pred
    # pred = np.zeros_like(pred)
    pred_corr = np.stack([bias_corr(pred[i, ...], ref[i, ...], mag=False)
                          for i in range(pred.shape[0])])  # loop over dynamics

    # Normalize # ToDo: How ideally normalize?
    ref_mag = np.abs(ref)
    pred_corr_mag = np.abs(pred_corr)
    ref_mag = (ref_mag - ref_mag.min()) / (ref_mag.max() - ref_mag.min())
    pred_corr_mag = (pred_corr_mag - pred_corr_mag.min()) / (ref_mag.max() - ref_mag.min()) # rescale to reference

    return pred_corr_mag, ref_mag

def log_quant_metrics(pred, ref):
    log_quant_results = {}

    for ech in range(pred.shape[1]):
        eval_dict = get_eval_metrics(pred[:, ech, ...],
                                     ref[:, ech, ...],
                                     axes = (1,2), mean = True)

        for key, value in eval_dict.items():
            if "std" not in key:
                log_quant_results[f'{key}{ech}'] = value

    return log_quant_results

def log_difference_images(pred_img, ref, mag_factor=1):
    log_results = {}

    if isinstance(pred_img, torch.Tensor):
        pred = pred_img.cpu().numpy()
    else:
        pred = pred_img

    assert pred.shape[1:] == ref.shape[1:]
    assert pred.shape[0] == ref.shape[0] or pred.shape[0] == ref.shape[0] // 2
    pred = create_hysteresis(pred) if pred.shape[0] == ref.shape[0] // 2 else pred
    # pred = np.zeros_like(pred)
    pred_corr = np.stack([bias_corr(pred[i, ...], ref[i, ...], mag=False)
                          for i in range(pred.shape[0])])  # loop over dynamics


    # ToDO: normalize pred_corr
    diff = np.abs(np.abs(pred_corr) - np.abs(ref))
    diff = (diff * mag_factor)
    # diff = (diff * mag_factor) / np.max(diff) * 255
    diff_int = np.clip(diff * 255, 0, 255).astype(np.uint8)
    fps = np.floor((pred_corr.shape[0]) / 5)

    for ech in range(diff.shape[1]):
        if diff.shape[0] > 1:
            log_results.update({
                "diff" + str(ech): wandb.Video(diff_int[:,ech,...], fps=fps, format="gif")
            })
            diff_static = medutils.visualization.plot_array(diff[::5, ech, ...].squeeze())
        else:
            log_results.update({"diff" + str(ech): wandb.Image(diff[:,ech,...])})
            diff_static = diff.squeeze()

        plot_difference(diff_static, vmin=0.0, vmax = 1.0, log_wandb=True, title="diff_static" + str(ech))

    return log_results

def log_recon_to_wandb(kpred_all, csm, ref=None, multi_coil=True, log_xt=True, log_diff=False):
    log_results = {}

    # k_min = np.min(kpred_all.detach().abs().cpu().numpy()) # find max of all echoes, to have same scale for all kspaces
    # k_max = np.max(kpred_all.detach().abs().cpu().numpy()) # find max of all echoes, to have same scale for all kspaces

    # for ech in range(kpred_all.shape[1]):
    #     kpred = kpred_all[:, ech, ...]
    vis_img = k2img_multiecho(kpred_all, csm=csm, scale=True, multi_coil=multi_coil)

    # ech = NIKmodel.config["echo"] if "echo" in NIKmodel.config and NIKmodel.config[
    #     "echo"] is not None else ech  # overwrite echo for single echo case1
    # log_results.update({
    #     'k' + str(ech): wandb.Video(vis_img['k_mag'], fps=1, format="gif"),
    #     'img' + str(ech): wandb.Video(vis_img['combined_mag'], fps=1, format="gif"),
    #     'img_phase' + str(ech): wandb.Video(vis_img['combined_phase'], fps=1, format="gif"),
    #     'khist' + str(ech): wandb.Histogram(torch.view_as_real(kpred).detach().cpu().numpy().flatten()),
    # })

    fps = np.floor((vis_img['k_mag_coil0'].shape[0]) / 5)

    for ech in range(kpred_all.shape[1]):

        from medutils.visualization import contrastStretching
        kpred_all = kpred_all

        if vis_img['k_mag_coil0'][:,ech,...].shape[0] > 1:
            wandb_log_type_func = lambda vid: wandb.Video(vid, fps=fps, format="gif")
            log_results.update({'img_phase' + str(ech): wandb_log_type_func(vis_img['combined_phase'][:,ech,...])})
        else:
            wandb_log_type_func = wandb.Image
            log_results.update({'img_phase' + str(ech): wandb_log_type_func(vis_img['combined_phase'][0, ech, ...].transpose(1,2,0))})

        log_results.update({
            'k' + str(ech): wandb_log_type_func(vis_img['k_mag_coil0'][:,ech,...]),
            'img' + str(ech): wandb_log_type_func(contrastStretching(vis_img['combined_mag'][:,ech,...])),
            'khist' + str(ech): wandb.Histogram(torch.view_as_real(kpred_all[:,ech,...]).detach().cpu().numpy().flatten()),
        })
        if multi_coil:
            log_results.update({
                'multicoil_k_mag' + str(ech): wandb_log_type_func(vis_img['multicoil_k_mag'][:, ech, ...]),
                'multicoil_img' + str(ech): wandb_log_type_func(vis_img['multicoil_mag'][:, ech, ...]),
                })

        if log_xt:
            # ToDo: expand
            idx = vis_img["combined_mag"].shape[-1] // 2 - 20
            img_marked = vis_img["combined_mag"].copy()
            img_marked[:,ech, :, :, int(idx)] = np.max(img_marked)
            log_results.update({
                'img_t_v': wandb.Image(img_marked[0, ech, 0, :, :]),
                'img_t': wandb.Image(contrastStretching(vis_img["combined_mag"][:, ech, 0, :, int(idx)]))
            })

        if log_diff:
            log_results.update({
                'img_t_v': wandb.Image(img_marked[0, ech, 0, :, :]),
                'img_t': wandb.Image(contrastStretching(vis_img["combined_mag"][:, ech, 0, :, int(idx)]))
            })

    return vis_img, log_results

def create_hysteresis(pred, flip_axis=0):
    if isinstance(pred, torch.Tensor):
        return torch.vstack([pred, torch.flip(pred, dims=[flip_axis])])
    else:
        return np.vstack((pred, np.flip(pred, axis=flip_axis)))

def bias_corr(img, ref, mag=True, norm=True):
    """
    Corrects bias between two complex images.

    Performs bias correction for two complex images using least squares regression.

    Parameters:
        img (numpy.ndarray): The input complex image to be corrected.
        ref (numpy.ndarray): The reference complex image to be used for correction.
        mag (bool, optional): Whether to use magnitude-only for correction (default is True).

    Returns:
        numpy.ndarray: The bias-corrected complex image.

    Note:
        If `mag` is set to True, the magnitude of the input and reference images will be used for correction.
        Otherwise, both the real and imaginary parts will be used.
    """
    if mag:
        ref = np.abs(ref)
        img = np.abs(img)

    im_size = img.shape
    i_flat = img.flatten()
    # i_flat = np.concatenate((i_flat.real, i_flat.imag))
    a = np.stack([i_flat, np.ones_like(i_flat)], axis=1)

    b = ref.flatten()
    # b = np.concatenate((b.real, b.imag))

    x = np.linalg.lstsq(a, b, rcond=None)[0]
    temp = img * x[0] + x[1]
    img_corr = np.reshape(temp, im_size)

    return img_corr


def get_eval_metrics(img, ref, axes=(1,2), mean=True):
    """
    Calculate evaluation metrics for an image compared to a reference image.

    Parameters:
        img (numpy.ndarray): The input image to be evaluated.
        ref (numpy.ndarray): The reference image to compare against.

    Returns:
        dict: A dictionary containing evaluation metrics (SSIM, PSNR, NRMSE, and NRMSE with absolute values).
    """
    metrics_dict = {
        "ssim": [ssim(img[i], ref[i], axes=axes) for i in range(img.shape[0])],
        "psnr": [psnr(img[i], ref[i], axes=axes) for i in range(img.shape[0])],
        "rmse": [nrmse(img[i], ref[i], axes=axes) for i in range(img.shape[0])],
        # "psnr_std": np.float32(np.round(np.std([psnr(img[i], ref[i]) for i in range(img.shape[0])]), 3)),
        # "nrmse": [nrmse(img[i] * (ref[i].max()/img[i].max()), ref[i], axes=axes) for i in range(img.shape[0])],
        "fsim": [fsim(img[i].squeeze()[...,None], ref[i].squeeze()[...,None]) for i in range(img.shape[0])],
    }

    # Temporal FSIM
    # phantom data limits : [70:155, 60:165]
    if img.shape[0] > 1:
        img_crop, ref_crop = img[:, :,70:155, 60:165], ref[:, :,70:155, 60:165]
        metrics_dict["fsim_xt"] = [fsim(img_crop[:,:,i,:].squeeze()[...,None], ref_crop[:,:,i,:].squeeze()[...,None]) for i in range(img_crop.shape[-2])]
        metrics_dict["fsim_yt"] = [fsim(img_crop[:,:,:,i].squeeze()[...,None], ref_crop[:,:,:,i].squeeze()[...,None]) for i in range(img_crop.shape[-2])]

    metrics_dict_all = {}
    for key in metrics_dict.keys():
        metrics_dict_all[key] = np.float32(round(np.mean(metrics_dict[key]), 3))
        metrics_dict_all[key + "_std"] = np.float32(round(np.std(metrics_dict[key]), 3))

    if mean:
        for key, value in metrics_dict_all.items():
            print(key, ":", value)

    return metrics_dict_all if mean else metrics_dict


def plot_difference(array, title, log_wandb=False, vmin=0.0, vmax=1.0):
    '''
    Plot three different views of 3D array
    '''

    fig = plt.figure(dpi=300)
    plt.imshow(array, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.colorbar()
    plt.title(title)
    if log_wandb:
        wandb.log({title: fig})
    else:
        plt.show()