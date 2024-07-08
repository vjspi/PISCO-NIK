import os
import time

import torch
import argparse
import numpy as np
import random
from pathlib import Path
import os
import imageio
import yaml
import io
import medutils
import glob
import json
import wandb

# import utils.eval
# from utils.basic import parse_config, import_module, find_subfolder
from torch.utils.data import DataLoader

from utils import basic
from utils.vis import angle2color, k2img, alpha2img, k2img_multiecho, \
    plot_kreg_weights, log_kspace_weights
from utils import eval
from utils.eval import log_recon_to_wandb, log_quant_metrics, log_difference_images, \
    postprocess_without_reference, postprocess_with_reference
from reference.xdgrasp import xdgrasp
from eval import test_sos_phantom, test_sos_subject

def main(config_input=None):
    if config_input is not None:
        config = config_input
    else:
        # parse args and get config
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', type=str, default='configs/config_abdominal.yml')
        parser.add_argument('-g', '--gpu', type=int, default=0)
        parser.add_argument('-r', '--r_acc', type=float, default=None)
        parser.add_argument('-sub', '--subject', type=str, default=None)
        parser.add_argument('-s', '--slice', type=int, default=None)
        parser.add_argument('-log', '--log', type=int, choices=[0, 1], default=1)
        parser.add_argument('-e', '--encoding', type=str, choices=["spatial", "STIFF"], default=None)
        parser.add_argument('-feat', '--hidden_features', type=int, default=None)
        parser.add_argument('-nav', '--nav_range', type=str, default=None)
        parser.add_argument('-ep', '--epochs', type=int, default=None)
        parser.add_argument('-a', '--alpha', type=float, default=None)
        parser.add_argument('-l', '--lamda', type=float, default=None)
        parser.add_argument('-od', '--overdetermination', type=float, default=None)


        parser.add_argument('-seed', '--seed', type=int, default=0)
        # parser.add_argument('-s', '--seed', type=int, default=0)
        args = parser.parse_args()

        # enable Double precision
        torch.set_default_dtype(torch.float32)

        # set gpu and random seed
        # torch.cuda.set_device(args.gpu)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)

        if torch.cuda.is_available():
            print("yes")
        else:
            print("no")

        num_cuda_devices = torch.cuda.device_count()
        if num_cuda_devices > 0:
            print("Found", num_cuda_devices, "CUDA device(s) available.")
            for i in range(num_cuda_devices):
                print("CUDA Device", i, ":", torch.cuda.get_device_name(i))
        else:
            print("No CUDA devices found.")

        # parse config
        config = basic.parse_config(args.config)
        if args.subject is not None:    # get general subject info
            data_config_path = os.path.join(os.getcwd(), "configs/subjects/", args.subject + ".yml")
            data_config = basic.parse_config(data_config_path)
            config.update(data_config)
            # config['subject_name'] = args.subject
        config['data_root'] = os.path.join(Path.home(), config["data_root"])
        config["results_root"] = os.path.join(Path.home(), config["results_root"])
        # config['slice_name'] = slice_name
        config['gpu'] = args.gpu
        torch.cuda.set_device(args.gpu)

        config['log'] = bool(args.log)
        # config['exp_summary'] = args.log

        # optional from command line (otherwise in config)
        if args.r_acc is not None:
            config['dataset']['acc_factor'] = int(args.r_acc) if args.r_acc == int(args.r_acc) else args.r_acc
        if args.slice is not None:
            config['slice'] = args.slice
        if args.encoding is not None:
            config["encoding"]["type"] = args.encoding
        if args.hidden_features is not None:
            config["model"]["params"]["hidden_features"] = args.hidden_features
        if args.epochs is not None:
            config["num_steps"] = args.epochs
        if args.nav_range is not None:
            config["dataset"]["nav_min"], config["dataset"]["nav_max"] = \
                tuple(map(float, args.nav_range.split(',')))
            # config["dataset"]["nav_min"] = args.nav[0]
            # config["dataset"]["nav_max"] = args.nav[1]

        # Kreg settings
        if args.lamda is not None:
            config["kreg"]["reg_lamda"] = args.lamda
        if args.alpha is not None:
            config["kreg"]["reg_alpha"] = args.alpha
        if args.overdetermination is not None:
            config["kreg"]["overdetermination"] = args.overdetermination


        config["exp_name"] = f"{config['exp_name']}_od{config['kreg']['overdetermination']}_lamda{config['kreg']['reg_lamda']}_alpha{config['kreg']['reg_alpha']}_" \
                             f"{config['encoding']['type']}" \
                             f"{config['model']['params']['hidden_features']}_" \
                             f"nav{config['dataset']['nav_min']}:{config['dataset']['nav_max']}"

    # create dataset
    dataset_class = basic.import_module(config["dataset"]["module"], config["dataset"]["name"])
    dataset = dataset_class(config)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                            num_workers=config['num_workers'], drop_last=True)
    config['nc'] = dataset.n_coils
    config['ne'] = dataset.n_echo if hasattr(dataset, "n_echo") else None
    config['nx'] = dataset.csm.shape[1]     # overwrite, since relevant in case data was downsampled
    config['ny'] = dataset.csm.shape[2]
    config['n_spokes'] = dataset.n_spokes

    ## Data debug
    # config["xdgrasp"]["beta"] = (0.1,0)
    # config["xdgrasp"]["lamdb"] = 0.1
    # xdgrasp(dataset.kdata[None,None,...], dataset.traj[None,None,...], dataset.self_nav,
    #         dataset.csm[..., config["slice"]], dataset.weights, config = config["xdgrasp"], ref = dataset.ref)
    # create model
    model_class = basic.import_module(config["model"]["module"], config["model"]["name"])
    NIKmodel = model_class(config)

    if config_input is not None:
        NIKmodel.model_save_path = config_input["model_save_path"]
        NIKmodel.results_save_path = config_input["results_save_path"]

    NIKmodel.init_train(resume=False) # Do init train before loading model!
    NIKmodel.coil_factors = dataset.coil_factors if hasattr(dataset, "coil_factors") else None
    # print params
    params = basic.count_parameters(NIKmodel.network_kdata)
    print("Network contains {} trainable parameters.".format(params))

    ### Load pretrained MLP
    if "pretrained" in NIKmodel.config["model"] and NIKmodel.config["model"]["pretrained"] is not None:
        start_epoch = None
        group_name = config["model"]["pretrained"]["pretrain_group"] + "_S" + str(config["subject_name"])
        exp_name = (config["model"]["pretrained"]["pretrain_exp"] + "*" + "_slice" + str(config["slice"]) + "*"
                    + "*_R" + str(config['dataset']['acc_factor']))
        if isinstance(NIKmodel.config["model"]["pretrained"]["epoch"], int):
            model_name = "_e{}".format(NIKmodel.config["model"]["pretrained"]["epoch"])
            start_epoch = NIKmodel.config["model"]["pretrained"]["epoch"] + 1
        else:
            model_name = NIKmodel.config["model"]["pretrained"]["epoch"]
        group_path = basic.find_subfolder(config["results_root"], group_name)
        exp_path = basic.find_subfolder(group_path, exp_name)
        # find latest model
        run_id = os.listdir(exp_path)[0]
        NIKmodel.config['weight_path'] = os.path.join(exp_path, run_id, "model_checkpoints", model_name)

        NIKmodel.load_network()
        if start_epoch is None:
            start_epoch = torch.load(NIKmodel.config['weight_path'],
                                 map_location=NIKmodel.device)["epoch"] + 1
        print("Continuing model training from epoch {}".format(start_epoch))
    else:
        start_epoch = 0


    # if "pretrained" in NIKmodel.config["model"] and NIKmodel.config["model"]["pretrained"] is not None:
    #     start_epoch = NIKmodel.config["model"]["pretrained"]["epoch"] + 1
    #     path = os.path.join(Path.home(), "{}/_e{}".format(NIKmodel.config["model"]["pretrained"]["path"],
    #                                                       NIKmodel.config["model"]["pretrained"]["epoch"]))
    #     NIKmodel.config['weight_path'] = path
    #     NIKmodel.load_network()
    #     # NIKmodel.to(torch.device(f'cuda:{args.gpu}'))
    #     # Ensure that the pretrained weights are on the same GPU
    #     # for key in NIKmodel.network_kdata.state_dict():
    #     #     NIKmodel.network_kdata.state_dict()[key] = NIKmodel.network_kdata.state_dict()[key].to(
    #     #         torch.device(f'cuda:{args.gpu}'))
    # else:
    #     start_epoch = 0


    # set log settings
    if config['exp_summary'] == 'wandb':
        # log params
        params_to_log = []
        for idx, (name, param) in enumerate(NIKmodel.named_parameters()):
            if "omega" in name:
                params_to_log.append(name)
        # wandb.watch(NIKmodel, params_to_log, log_graph=True)
        # wandb.watch(NIKmodel, "all", log_graph=False)

    # save config for later evaluation
    NIKmodel.config["model_save_path"] = NIKmodel.model_save_path
    NIKmodel.config["result_save_path"] = NIKmodel.result_save_path
    NIKmodel.config["weight_path"] = NIKmodel.weight_path
    if os.path.exists(NIKmodel.model_save_path):
        with io.open(NIKmodel.model_save_path + '/config.yml', 'w', encoding='utf8') as outfile:
            yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True)

    loss_model = 1e10
    loss_reg_flag = True
    # from torchinfo import summary
    # summary(NIKmodel.network_kdata, (1, NIKmodel.config["feature_dim"]))

    start_time = time.time()
    #% Train model
    for epoch in range(0, config['num_steps']):

        if epoch < start_epoch:
            continue
        # if epoch == 100:
        #     NIKmodel.config["kreg"]["optim_type"] = "joint_noBack"

        # DC loss
        loss_epoch, loss_dc_epoch, loss_reg_epoch = 0,0,0
        W_reg = []


        for i, sample in enumerate(dataloader):
            loss, [loss_dc, loss_reg], W_reg_i = NIKmodel.train_batch(sample)
            W_reg.append(W_reg_i)
            print(f"Epoch: {epoch}, Iter: {i}, Loss: {loss}, Loss_kreg: {loss_reg}")
            loss_epoch += loss
            loss_dc_epoch += loss_dc
            loss_reg_epoch += loss_reg

        log_dict = {
            'epoch': epoch,
            'loss': loss_epoch.item() / len(dataloader),
            'loss_dc': loss_dc_epoch.item() / len(dataloader),
            'loss_reg': loss_reg_epoch.item() / len(dataloader)
        }

        if epoch % 100 ==0:
            duration = time.time() - start_time
            with open(NIKmodel.result_save_path + "duration.txt", "w") as file:
                file.write(f"Training duration {epoch}: {duration} seconds. Log: {config['log_test']}")

        # log test reconstruction
        if 'log_test' in config and config["log_test"] and 'log' in config and config["log"]:

            if epoch % 10 == 0:
                ## Log weights
                if W_reg[0] is not None:
                    # Determine the maximum size
                    # # Pad arrays to match the maximum size
                    # max_set_size = [max([W_reg[i][idx].shape for i in range(len(W_reg))]) for idx in range(len(W_reg[0]))]
                    # W_reg = [[torch.nn.functional.pad(W_reg[i][idx], (0, max_set_size[idx][0] - W_reg[i][idx].shape[0])) for
                    #      i in range(len(W_reg))] for idx in range(len(W_reg[0]))]
                    # W_reg = [torch.stack([W_reg[i][idx] for i in range(len(W_reg))], dim=0) for idx in range(len(W_reg[0]))]

                    W_reg = [torch.stack([W_reg[i][idx] for i in range(len(W_reg))], dim=0) for idx in range(len(W_reg[0]))]
                    # W_mean_dict = log_kspace_weights(W_reg[0].detach(), title="W_Mean", vmin=0, vmax=1.0)
                    # W_shift_dict = log_kspace_weights(W_reg[1].detach(), title="W_Shift", vmin=0, vmax=1.0)
                    W_dict = log_kspace_weights(W_reg[2].detach(), title="W", vmin=0, vmax=1.0)
                    # log_dict.update(W_mean_dict)
                    # log_dict.update(W_shift_dict)
                    log_dict.update(W_dict)

                # kpred is of size c*t*y*x - y and x are defined by the size of csm (see above)
                config["nt"] = 50 if (np.min(dataset.self_nav) != np.max(dataset.self_nav) and config["dataset"]["nav_min"] != config["dataset"]["nav_max"]) else 1

                kpred_all = NIKmodel.test_batch(input_dim=[config["nt"], config["nx"], config["ny"]])
                kpred_all = kpred_all.unsqueeze(1) if len(kpred_all.shape) == 4 else kpred_all  # add echo dimension

                vis_img, temp_dict = log_recon_to_wandb(kpred_all, csm = dataset.csm[..., NIKmodel.config["slice"]],
                                               multi_coil=True, log_xt=True)
                log_dict.update(temp_dict)

                # Quantitative:
                if "phantom" in str(NIKmodel.config["subject_name"]):
                    ref = dataset.ref.transpose(3, 4, 2, 0, 1)
                    ref = medutils.visualization.normalize(ref, max_int=1)
                    img = vis_img["combined_img"]
                    img = eval.create_hystereses(img, dim_axis=0)
                    img = eval.postprocess(img, ref=ref)
                    # img_mag, ref_mag = postprocess_with_reference(vis_img["combined_img"], dataset.ref.transpose(3, 4, 2, 0, 1))
                    eval_dict = log_quant_metrics(img, ref)
                    diff_dict = log_difference_images(img, ref)
                    log_dict.update(eval_dict)
                    log_dict.update(diff_dict)

            # log progress
            NIKmodel.exp_summary_log(log_dict, step=epoch)

        # save checkpoints
        if 'log' in config and config["log"]:
            if loss_model > loss_epoch:
                l = loss_epoch / len(dataloader)
                NIKmodel.save_network("best_network", epoch, l.detach().item())
                loss_model = loss_epoch
                # save best images

            loss_reg_e = loss_reg_epoch / len(dataloader)
            if loss_reg_e < 0.1 and loss_reg_flag == True:
                NIKmodel.save_network("lossreg0.1", epoch, loss_reg.detach().item())
                loss_reg_flag = False

            if epoch % 50 == 0:
                l = loss_epoch / len(dataloader)
                NIKmodel.save_network("_e{}".format(epoch), epoch, l.detach().item())
                # save images
                # middle index
                # if hasattr(NIKmodel, 'log_test') and config["log_test"]:
                #     t = int(np.floor(vis_img["combined_mag"].shape[0] / 2))
                #     imageio.imwrite(NIKmodel.model_save_path + '/recon_middlnav_e{}.png'.format(epoch),
                #                     vis_img["combined_mag"][t, ...].squeeze())

if __name__ == '__main__':
    main()