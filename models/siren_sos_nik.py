import torch
import torch.nn as nn
import numpy as np
import os

from utils.mri import coilcombine, ifft2c_mri
from .base import NIKBase
from models.base_sos import NIKSoSBase
from utils.basic import import_module
import models.layers as layers

class NIKSiren(NIKSoSBase):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.create_network()
        self.to(self.device)

    def create_network(self):
        # overwrite outdim with number of channels
        self.config["out_dim"] = self.config["nc"]
        out_dim = self.config["out_dim"]
        coord_dim = self.config["coord_dim"]
        self.config['model']['params']["bias_last"] = self.config['model']['params']["bias_last"] if "bias_last" in self.config['model']['params'] else True
        self.config['model']['params']["bias_hidden"] = self.config['model']['params']["bias_hidden"] if "bias_hidden" in self.config['model']['params'] else True

        if "params" in self.config["model"]:
            self.network_kdata = Siren(coord_dim, out_dim,
                                       **self.config['model']['params']).to(self.device)
        else:
            feature_dim = self.config["feature_dim"]
            num_layers = self.config["num_layers"]
            omega = self.config["omega"]
            self.network_kdata = Siren(coord_dim, out_dim, feature_dim, num_layers,
                                      omega_0=omega).to(self.device)

        if "encoding" in self.config:
            self.network_kdata.generate_B(self.config["encoding"])
        else:
            self.network_kdata.generate_B(out_dim)

        # self.network_kdata = Siren(coord_dim, feature_dim, num_layers, out_dim,
        #                           omega_0=omega).to(self.device)

    def init_expsummary(self):
        """
        Initialize the visualization tools.
        Should be called in init_train after the initialization of self.exp_id.
        """
        if self.config['exp_summary'] == 'wandb':
            import wandb
            self.exp_summary = wandb.init(
                project=self.config['wandb_project'],
                name=self.exp_id,
                config=self.config,
                group=self.group_id,
                entity=self.config['wandb_entity']
            )

    def init_train(self):
        """Initialize the network for training.
        Should be called before training.
        It does the following things:
            1. set the network to train mode
            2. create the optimizer to self.optimizer
            3. create the model save directory
            4. initialize the visualization tools
        If you want to add more things, you can override this function.
        """
        self.network_kdata.train()

        self.create_criterion()
        self.create_optimizer()

        self.load_names()

        if 'log' in self.config and self.config["log"]:
            self.init_expsummary()
            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)


    def init_test(self):
        """Initialize the network for testing.
        Should be called before testing.
        It does the following things:f
            1. set the network to eval mode
            2. load the network parameters from the weight file path
        If you want to add more things, you can override this function.
        """
        self.weight_path = self.config['weight_path']

        self.load_network()

        self.network_kdata.eval_dict()

        exp_id = self.weight_path.split('/')[-2]
        epoch_id = self.weight_path.split('/')[-1].split('.')[0]

        # setup model save dir
        results_save_dir = os.path.join(self.group_id, self.exp_id)
        if "results_root" in self.config:
            results_save_dir = "".join([self.config["results_root"],'/', results_save_dir])

        if not os.path.exists(results_save_dir):
            os.makedirs(results_save_dir)

        self.result_save_path

    def pre_process(self, inputs):
        """
        Preprocess the input coordinates.
        """

        inputs['coords'] = inputs['coords'].to(self.device)     # required for loss calculation

        # features = self.network_kdata.pre_process(inputs['coords_patch'])

        if "encoding" in self.config:
            if self.config["encoding"]["type"] == "spatial":
                factor = 2 * torch.pi if "pi_factor" in self.config["encoding"] and self.config["encoding"]["pi_factor"] else 1.
                features = torch.cat([torch.sin(factor * inputs['coords'] @ self.network_kdata.B),
                                      torch.cos(factor * inputs['coords'] @ self.network_kdata.B)], dim=-1)
            elif self.config["encoding"]["type"] == "spatial+temporal":
                factor = 2 * torch.pi if ["pi_factor"] in self.config["encoding"] and self.config["encoding"][
                    "pi_factor"] else 1.
                features = torch.cat([torch.sin(factor * inputs['coords'][:, 1:] @ self.network_kdata.B_spat),
                                      torch.cos(factor * inputs['coords'][:, 1:] @ self.network_kdata.B_spat),
                                      torch.sin(factor * inputs['coords'][:, :1] @ self.network_kdata.B_temp),
                                      torch.cos(factor * inputs['coords'][:, :1] @ self.network_kdata.B_temp)],
                                     dim=-1)
            elif self.config["encoding"]["type"] == "STIFF":
                factor = 2 * torch.pi if "pi_factor" in self.config["encoding"] and self.config["encoding"][
                    "pi_factor"] else 1.
                features = torch.cat([torch.cos(factor * inputs['coords'][:, 1:] @ self.network_kdata.B_spat),
                                      torch.sin(factor * inputs['coords'][:, 1:] @ self.network_kdata.B_spat),
                                      torch.cos(factor * inputs['coords'][:, 1:] @ self.network_kdata.B_temp) * torch.cos(2 * torch.pi * inputs['coords'][:, :1]),
                                      torch.cos(factor * inputs['coords'][:, 1:] @ self.network_kdata.B_temp) * torch.sin(2 * torch.pi * inputs['coords'][:, :1]),
                                      torch.sin(factor * inputs['coords'][:, 1:] @ self.network_kdata.B_temp) * torch.cos(2 * torch.pi * inputs['coords'][:, :1]),
                                      torch.sin(factor * inputs['coords'][:, 1:] @ self.network_kdata.B_temp) * torch.sin(2 * torch.pi * inputs['coords'][:, :1])
                                      ], dim=-1)

        else:
            features = torch.cat([torch.sin(inputs['coords'] @ self.network_kdata.B),
                                  torch.cos(inputs['coords'] @ self.network_kdata.B)], dim=-1)
        # else:
        #     inputs['coords'] = inputs['coords'].to(self.device)  # required for loss calculation
        #     features = self.network_kdata.pre_process(inputs['coords'])

        inputs['features'] = features

        if inputs.keys().__contains__('targets'):
            inputs['targets'] = inputs['targets'].to(self.device)

        return inputs

    def post_process(self, output):
        """
        Convert the real output to a complex-valued output.
        The first half of the output is the real part, and the second half is the imaginary part.
        """
        output = torch.complex(output[..., 0:self.config["out_dim"]], output[..., self.config["out_dim"]:])
        return output

    def train_batch(self, sample):

        self.network_kdata.train()
        # self.conv_patch.train() if hasattr(self, 'conv_patch') else None

        self.optimizer.zero_grad()

        sample = self.pre_process(sample)
        output = self.forward(sample)
        output = self.post_process(output)
        loss, reg = self.criterion(output, sample['targets'], sample['coords'])
        loss.backward()
        self.optimizer.step()
        return loss

    def test_batch(self, input=None, input_dim=None):
        """
        Test the network with a cartesian grid.
        if sample is not None, it will return image combined with coil sensitivity.
        """
        self.network_kdata.eval_dict()
        self.conv_patch.eval_dict() if hasattr(self, 'conv_patch') else None

        with torch.no_grad():

            nc = self.config['nc']  # len(self.config['coil_select'])  # nc = self.config['nc']

            if input is None:

                if input_dim is not None:
                    assert len(input_dim) == self.config["coord_dim"]
                    nnav, nx, ny = input_dim
                else:
                    nx = self.config['nx']
                    ny = self.config['ny']
                    nnav = self.config['nnav']

                nav_min = self.config["dataset"]["nav_min"] if "nav_min" in self.config["dataset"] else -1.
                nav_max = self.config["dataset"]["nav_max"] if "nav_max" in self.config["dataset"] else -1.
                delta_nav = (nav_max - nav_min) / nnav

                # coordinates: contr, kx, ky, nc, nav
                # contrs = torch.linspace(-1, 1, ncontr)
                # kc = torch.linspace(-1, 1, nc)
                kxs = torch.linspace(-1, 1 - 2 / nx, nx)
                kys = torch.linspace(-1, 1 - 2 / ny, ny)
                # knav = torch.linspace(self.config["dataset"]["navigator_min"], 1, nnav)
                knav = torch.linspace(nav_min + delta_nav / nnav, nav_max - delta_nav / nnav, nnav)

                # TODO: disgard the outside coordinates before prediction
                grid_coords = torch.stack(torch.meshgrid(knav, kys, kxs, indexing='ij'), -1)  #  nav, nx, ny,3
                # grid_coords = grid_coords.to(self.device)
                dist_to_center = torch.sqrt(grid_coords[..., 1] ** 2 + grid_coords[..., 2] ** 2)
                dist_to_center = dist_to_center.unsqueeze(1).expand(-1, nc, -1, -1)  # nt, nc, nx, ny

                nDim = grid_coords.shape[-1]
                contr_split = 1

            else:
                grid_coords = input
                # grid_coords = grid_coords.to(self.device)
                nDim = grid_coords.shape[-1]
                contr_split = 1

            # split t for memory saving
            contr_split_num = np.ceil(grid_coords.shape[0] / contr_split).astype(int)

            kpred_list = []
            for t_batch in range(contr_split_num):
                grid_coords_batch = grid_coords[t_batch * contr_split:(t_batch + 1) * contr_split]

                grid_coords_batch = grid_coords_batch.reshape(-1, nDim).requires_grad_(False)
                # get prediction
                sample = {'coords': grid_coords_batch}
                sample = self.pre_process(sample)   # encode time differently?
                kpred = self.forward(sample)

                kpred = self.post_process(kpred)
                kpred_list.append(kpred)
            kpred = torch.concat(kpred_list, 0)

            # kpred_list.append(kpred)
            # kpred = torch.mean(torch.stack(kpred_list, 0), 0) #* filter_value.reshape(-1, 1)

            if input is None or input_dim is not None:  # reshape if default or input_dim given (but not if coords)
                # TODO: clearning this part of code
                kpred = kpred.reshape(nnav, ny, nx, nc).permute(0,3,1,2)
                k_outer = 1
                kpred[dist_to_center >= k_outer] = 0
                # kpred = kpred.permute(3, 0, 1, 2)  # coil dimension second, imgDim last
                # kpred = kpred.squeeze(-1)

            if hasattr(self, "coil_factors") and self.coil_factors is not None:
                # rescale coil based on maximum intensity
                kpred = kpred * torch.tensor(self.coil_factors.reshape(1,-1,1,1)).to(kpred)

            return kpred

    def forward(self, inputs):
        return self.network_kdata(inputs['features'])


"""
The following code is a demo of mlp with sine activation function.
We suggest to only use the mlp model class to do the very specific 
mlp task: takes a feature vector and outputs a vector. The encoding 
and post-process of the input coordinates and output should be done 
outside of the mlp model (e.g. in the prepocess and postprocess 
function in your NIK model class).
"""

class Siren(nn.Module):
    def __init__(self, coord_dim, out_dim, hidden_features, num_layers, omega_0=30, bias_hidden=True, bias_last=True, norm = None, exp_out=True) -> None:
        super().__init__()

        self.coord_dim = coord_dim
        self.hidden_features = hidden_features

        if norm == "weight":
            weight_norm = True
            norm = "none"
        else:
            weight_norm = False

        self.net = []
        first_layer = SineLayer(hidden_features, hidden_features,
                              is_first=True, bias=bias_hidden,
                              weight_norm=weight_norm, omega_0=omega_0)
        first_layer = nn.utils.weight_norm(first_layer) if weight_norm else first_layer # ToDo
        self.net.append(first_layer)
        self.net.append(layers.get_norm_1d(norm, hidden_features))
        for i in range(num_layers - 1):
            layer = SineLayer(hidden_features, hidden_features, bias=bias_hidden,
                                      is_first=False, weight_norm=weight_norm, omega_0=omega_0)
            layer = nn.utils.weight_norm(self.linear, name="weight", dim=None) if weight_norm else layer
            self.net.append(layer)
            self.net.append(layers.get_norm_1d(norm, hidden_features))

        final_linear = nn.Linear(hidden_features, out_dim * 2, bias=bias_last)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / omega_0,
                                         np.sqrt(6 / hidden_features) / omega_0)
        self.net.append(final_linear)
        self.net = nn.Sequential(*[layer for layer in self.net if layer is not None])

        torch.manual_seed(0)

    def generate_B(self, config=None):

        if config is None:
            config["type"] = "spatial"
            config["sigma"] = 1.

        if config["type"] == "spatial":
            B = torch.randn((self.coord_dim, self.hidden_features // 2), dtype=torch.float32) * config["sigma"]
            self.register_buffer('B', B)
        elif config["type"] == "spatial+temporal":
            spat_feat_num = int(config["spat_feat_perc"] * self.hidden_features)
            B_spat = torch.randn((self.coord_dim-1, (spat_feat_num) // 2 ),
                                 dtype=torch.float32) * config["sigma"]
            B_temp = torch.randn((1, (self.hidden_features - spat_feat_num) // 2), dtype=torch.float32) * config["sigma"]
            self.register_buffer('B_spat', B_spat)
            self.register_buffer('B_temp',B_temp)
        elif config["type"] == "STIFF":
            # static and dynamic component for k-space (both temporal & spatial apply to coordinates and t is multiplied in addition)
            spat_feat_num = int(config["spat_feat_perc"] * self.hidden_features)
            B_spat = torch.randn((self.coord_dim-1, (spat_feat_num) // 2 ),
                                 dtype=torch.float32) * config["sigma"]
            B_temp = torch.randn((self.coord_dim-1, (self.hidden_features - spat_feat_num) // 4), dtype=torch.float32) * config["sigma"]
            self.register_buffer('B_spat', B_spat)
            self.register_buffer('B_temp',B_temp)

    def forward(self, features):
        return self.net(features)


class SineLayer(nn.Module):
    """Linear layer with sine activation. Adapted from Siren repo"""

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, weight_norm=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # if weight_norm:
        #     self.linear = nn.utils.weight_norm(self.linear, name="weight", dim=None)


        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
