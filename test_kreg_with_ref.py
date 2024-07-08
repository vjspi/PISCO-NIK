import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import json
from pathlib import Path

import utils.vis
from utils import basic
import numpy as np
import medutils


def main(path, device=0, gif=False, overwrite=False):
    # Specifiy experiment of iconik
    exp_name = path
    results_path = exp_name + "/rec_test"
    phantom = True
    config = basic.parse_config(exp_name + "/model_checkpoints/config.yml")
    sub = config["subject_name"]
    slice = config["slice"]
    if "phantom" in sub:
        if "0.33FS" in sub:
           acc_factor = 3
        elif "0.5FS" in sub:
           acc_factor = 2
        elif "1FS" in sub:
           acc_factor = 1
    else:
        acc_factor = config["dataset"]["acc_factor"]
    model = np.load(exp_name + '/rec_test/recon_best_network.npz')

    # check if model is a kreg method
    if "pretrained" not in config["model"]:
        return
    else:
        pass

    if os.path.exists(results_path + '/eval.txt') and not overwrite:
        print("Not overwriting {}".format(results_path))
        return

    ### grasp reference
    grasp_path = config["data_root"]
    grasp = np.load(grasp_path + "S" + str(config["subject_name"]) + "/grasp_reference_{}.npz".format(slice), allow_pickle=True)

    ### load reference NIK
    nik_group_name = config["model"]["pretrained"]["pretrain_group"] + "_S" + str(config["subject_name"])
    nik_exp_name = config["model"]["pretrained"]["pretrain_exp"] + "*" + "_slice" + str(config["slice"]) + "_R" + str(config['dataset']['acc_factor'])
    nik_group_path = basic.find_subfolder(config["results_root"], nik_group_name)
    nik_exp_path = basic.find_subfolder(nik_group_path, nik_exp_name)
    nik_path = os.listdir(nik_exp_path)[0]
    nik = np.load(nik_exp_path + "/" + nik_path + '/rec_test/recon_best_network.npz', allow_pickle=True)
    # identify

     ## load data
    img = {}
    eval_dict ={}
    img["model"] = model["recon"]
    img["nik"] = nik["recon"]
    img["inufft"] = grasp["R1"].item()["INUFFTnufft"] if "phantom" in sub else grasp["R{}".format(acc_factor)].item()["INUFFTnufft"]
    img["xdgrasp4"] = grasp["R1"].item()["4MSgrasp"] if "phantom" in sub else grasp["R{}".format(acc_factor)].item()["4MSgrasp"]
    img["nufft4"] = grasp["R1"].item()["4MSnufft"] if "phantom" in sub else grasp["R{}".format(acc_factor)].item()["4MSnufft"]
    img["xdgrasp50"] = grasp["R1"].item()["50MSgrasp"] if "phantom" in sub else grasp["R{}".format(acc_factor)].item()["50MSgrasp"]
    img["nufft50"] = grasp["R1"].item()["50MSnufft"] if "phantom" in sub else grasp["R{}".format(acc_factor)].item()["50MSnufft"]
    if "phantom" in sub:
        img["ref"] = model["ref"].transpose(3,4,2,0,1)
    elif "knee" in sub:
        img["ref"] = grasp["R1"].item()["INUFFTnufft"].repeat(100,1,1,1,1) #  grasp["R{}".format(acc_factor)].item()["INUFFTnufft"].repeat(100,1,1,1,1)
    elif "11_R0" in sub:
        ref_path = config["data_root"]
        ref = np.load(ref_path + "S11_gated" + "/grasp_reference_{}.npz".format(slice), allow_pickle=True)
        img["ref"] = ref["R1"].item()["INUFFTnufft"].repeat(100,1,1,1,1)
    else:
        img["ref"] = grasp["R1"].item()["4MSgrasp"]


    ### Process images and calculate metrics
    import utils.eval as eval
    import utils.vis as vis
    ech = 0
    t = 10
    img["ref"] = basic.torch2numpy(img["ref"])
    img["ref"] = medutils.visualization.contrastStretching(img["ref"])
    img["ref"] = medutils.visualization.normalize(img["ref"], max_int=1)
    eval_str_xy, eval_str_xt, eval_str_yt = {},{},{}
    for key in img.keys():
        if "ref" not in key:
            # expand images to GT motion states
            img[key] = basic.torch2numpy(img[key])
            img[key] = eval.create_hystereses(img[key], dim_axis=0)
            img[key] = eval.postprocess(img[key], img["ref"])
            if "knee" in sub or "phantom" in sub:
                eval_dict[key] = eval.get_eval_metrics(img[key][:,ech, ...], img["ref"][:, ech,  ...])
            elif "11_R0" in sub:
                eval_dict[key] = eval.get_eval_metrics(img[key][[0], ech, ...], img["ref"][[t], ech, ...]) # calculate metric only for temproal value
            eval_str_xy[key] = eval.make_string_from_value_dict(eval_dict[key], default_keys=["psnr", "fsim"])
            eval_str_xt[key] = eval.make_string_from_value_dict(eval_dict[key], default_keys=["psnr", "fsim_xt"])
            eval_str_yt[key] = eval.make_string_from_value_dict(eval_dict[key], default_keys=["psnr", "fsim_yt"])


    ## ToDo: save comparison images

    # NUFFT, GRASP4, GRASP5, NIK, ICoNIK
    if "knee" in sub:
        plot_order = ["inufft", "nik", "model"]
    # elif "11_R0" in sub:
    #     plot_order = ["ref", "inufft", "nik", "model"]
    else:
        plot_order = ["ref", "nufft4", "xdgrasp4","xdgrasp50", "nik", "model"]


    img_sat = {}
    img_sat_max_list=[]
    for key in plot_order:
        img_sat[key] = medutils.visualization.contrastStretching(img[key][...].squeeze(), saturated_pixel=0.015)
        img_sat_max_list.append(img_sat[key].max())

    # Plot Spatial images
    plot_metrics = ["psnr", "fsim"]
    img_xy_list, img_max_list, title_list, eval_list = [], [], [],[]
    for key, entry in img_sat.items():
        img_xy_list.append(entry[t,...].squeeze())
        title_list.append(key)
        if key != "ref":
            eval_list.append(eval_str_xy[key])
        else:
            eval_list.append("")


    zoom_region = (105, 145, 50, 90) if "knee" in sub else (135,185,70,120)

    eval_list = eval_list if "R0" not in sub else None
    vis.plot_grid_from_lists([img_xy_list], [img_sat_max_list], [eval_list], [title_list],
                             zoom_region=None,
                             path=results_path + "/recon_comp_xy_t{}.jpg".format(t))
    vis.plot_grid_from_lists([img_xy_list], [img_sat_max_list], [eval_list], [title_list],
                             zoom_region=None,
                             path=results_path + "/recon_comp_xy_t{}.eps".format(t))
    vis.plot_grid_from_lists([img_xy_list], [img_sat_max_list], [eval_list], None,
                             zoom_region=zoom_region, crop_vert=45,
                             path=results_path + "/recon_comp_xy_t{}_zoom.jpg".format(t))
    vis.plot_grid_from_lists([img_xy_list], [img_sat_max_list], [eval_list], None,
                             zoom_region=zoom_region, crop_vert=45,
                             path=results_path + "/recon_comp_xy_t{}_zoom.eps".format(t))


    # Plot Spatial images
    t=5
    img_xy_list, title_list, eval_list = [], [],[]
    for key, entry in img_sat.items():
        # temp_img = medutils.visualization.contrastStretching(img[key][t,...].squeeze(), saturated_pixel=0.015)
        img_xy_list.append(entry[t,...].squeeze())
        # img_max_list.append(temp_img.max())
        title_list.append(key)
        if key != "ref":
            eval_list.append(eval_str_xy[key])
        else:
            eval_list.append("")


    eval_list = eval_list if "R0" not in sub else None
    vis.plot_grid_from_lists([img_xy_list], [img_sat_max_list], [eval_list], [title_list],
                             path=results_path + "/recon_comp_xy_t{}_.jpg".format(t))



    ### Plot Temporal images
    y = 160
    eval_metrics = ["fsim_xt"]
    plt.imshow(img["ref"][0, ..., :y, :].squeeze(), cmap="gray")
    plt.title("cut-line")
    plt.show()
    img_yt_list, title_list, eval_list = [], [],[]
    for key, entry in img_sat.items():
        img_yt_list.append(entry[:50,...,y,:].squeeze())
        title_list.append(key)
        if key != "ref":
            eval_list.append(eval_str_xt[key])
        else:
            eval_list.append("")


    eval_list = eval_list if "R0" not in sub else None
    vis.plot_grid_from_lists([img_yt_list],[img_sat_max_list], eval_list = None, title_list=[title_list], zoom_region=None,
                             path=results_path + "/recon_comp_xt_y{}.jpg".format(y))

    ### Plot Temporal images
    x = 75
    plt.imshow(img["ref"][0, ..., :, :x].squeeze(), cmap="gray")
    plt.title("cut-line")
    plt.savefig(results_path + "/recon_comp_yt_x{}_cutline.jpg".format(x))
    plt.show()
    img_yt_list, title_list, eval_list = [], [],[]
    for key, entry in img_sat.items():
        img_yt_list.append(entry[:50,...,:,x].squeeze())
        title_list.append(key)
        if key != "ref":
            eval_list.append(eval_str_yt[key])
        else:
            eval_list.append("")

    eval_list = eval_list if "R0" not in sub else None
    vis.plot_grid_from_lists([img_yt_list], [img_sat_max_list], eval_list = None, title_list=[title_list],
                             path=results_path + "/recon_comp_yt_x{}.jpg".format(x))

    if gif:
        eval_str = eval_str_xy if "R0" not in sub else None
        total_duration = 5
        knav = np.linspace(0, total_duration, img_sat["ref"].shape[0])
        for img_name, img_value in img_sat.items():
            utils.vis.save_gif(img_value, #str=eval_str[img_name],
                               numbers_array=np.around(knav, decimals=2),
                               filename=results_path + "/dyn_recon_{}.gif".format(img_name),
                               intensity_factor=1, total_duration=total_duration)

    with open(results_path + '/eval.txt', 'w') as f:
        json.dump(eval_dict, f, indent=4, default=basic.float32_serializer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='configs/config_abdominal.yml')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-o', '--overwrite', type=str, default="true")
    args = parser.parse_args()
    # Manually interpret the string as a boolean value
    args.overwrite = args.overwrite.lower() == 'true'
    main(path=args.path, device=args.gpu, overwrite=args.overwrite)

