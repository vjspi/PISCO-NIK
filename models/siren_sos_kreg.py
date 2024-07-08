import torch
import torch.nn as nn
import numpy as np
import os

import losses.hdr
from utils.mri import coilcombine, ifft2c_mri
from .base import NIKBase
from models.base_sos import NIKSoSBase
from utils.basic import import_module
from losses.hdr import HDRLoss_FF
from losses.pisco import PiscoLoss
from utils import kernel_coords

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

    def init_expsummary(self, resume=False):
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
                entity=self.config['wandb_entity'],
                resume=resume
            )

    def init_train(self, resume = False):
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
        self.create_regularizer()
        self.load_names()
        if 'log' in self.config and self.config["log"]:
            self.init_expsummary(resume=resume)
            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)

    def create_criterion(self):
        """Create the loss function."""
        self.criterion = HDRLoss_FF(self.config)
        # self.criterion = AdaptiveHDRLoss(self.config)
    def create_optimizer(self):
        """Create the optimizer."""
        # self.optimizer = torch.optim.Adam([self.parameters(), self.network.parameters()], lr=self.config['lr'])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config['lr']))

    def create_regularizer(self):
        self.criterion_reg = PiscoLoss(self.config)

        ## Create regularization optimizer
        if self.criterion_reg.optim == "SGD":
            self.optimizer_reg = torch.optim.SGD(self.parameters(), lr=self.criterion_reg.lr)
        elif self.criterion_reg.optim == "Adam":
            self.optimizer_reg = torch.optim.Adam(self.parameters(), lr=self.criterion_reg.lr)
        elif self.criterion_reg.optim == "RMSProp":
            self.optimizer_reg = torch.optim.RMSprop(self.parameters(), lr=self.criterion_reg.lr)
        else:
            AssertionError("Optimizer for criterion_reg not defined")

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

        self.network_kdata.eval()

        exp_id = self.weight_path.split('/')[-2]
        epoch_id = self.weight_path.split('/')[-1].split('.')[0]

        # setup model save dir
        results_save_dir = os.path.join(self.group_id, self.exp_id)
        if "results_root" in self.config:
            results_save_dir = "".join([self.config["results_root"],'/', results_save_dir])

        if not os.path.exists(results_save_dir):
            os.makedirs(results_save_dir)

        self.result_save_path

    def pre_process(self, coords):
        """
        Preprocess the input coordinates.
        """

        # inputs['coords'] = inputs['coords'].to(self.device)     # required for loss calculation
        coords = coords.to(self.device)     # required for loss calculation

        # features = self.network_kdata.pre_process(inputs['coords_patch'])

        if "encoding" in self.config:
            if self.config["encoding"]["type"] == "spatial":
                factor = 2 * torch.pi if "pi_factor" in self.config["encoding"] and self.config["encoding"]["pi_factor"] else 1.
                features = torch.cat([torch.sin(factor * coords @ self.network_kdata.B),
                                      torch.cos(factor * coords @ self.network_kdata.B)], dim=-1)
            elif self.config["encoding"]["type"] == "spatial+temporal":
                factor = 2 * torch.pi if ["pi_factor"] in self.config["encoding"] and self.config["encoding"][
                    "pi_factor"] else 1.
                features = torch.cat([torch.sin(factor * coords[:, 1:] @ self.network_kdata.B_spat),
                                      torch.cos(factor * coords[:, 1:] @ self.network_kdata.B_spat),
                                      torch.sin(factor * coords[:, :1] @ self.network_kdata.B_temp),
                                      torch.cos(factor * coords[:, :1] @ self.network_kdata.B_temp)],
                                     dim=-1)
            elif self.config["encoding"]["type"] == "STIFF":
                factor = 2 * torch.pi if "pi_factor" in self.config["encoding"] and self.config["encoding"][
                    "pi_factor"] else 1.
                features = torch.cat([torch.cos(factor * coords[:, 1:] @ self.network_kdata.B_spat),
                                      torch.sin(factor * coords[:, 1:] @ self.network_kdata.B_spat),
                                      torch.cos(factor * coords[:, 1:] @ self.network_kdata.B_temp) * torch.cos(2 * torch.pi * coords[:, :1]),
                                      torch.cos(factor * coords[:, 1:] @ self.network_kdata.B_temp) * torch.sin(2 * torch.pi * coords[:, :1]),
                                      torch.sin(factor * coords[:, 1:] @ self.network_kdata.B_temp) * torch.cos(2 * torch.pi * coords[:, :1]),
                                      torch.sin(factor * coords[:, 1:] @ self.network_kdata.B_temp) * torch.sin(2 * torch.pi * coords[:, :1])
                                      ], dim=-1)

        else:
            features = torch.cat([torch.sin(coords @ self.network_kdata.B),
                                  torch.cos(coords @ self.network_kdata.B)], dim=-1)

        return features


    # def get_radial_kernel(self):
    #     # tbd
    #     # determine angle to next phase encoding line
    #     dphi = torch.pi / self.config["n_spokes"] # ToDo: Check
    # def get_cartesian_kernel(self):
    #     #tbd


    def sample_patches(self, sample, requires_grad = False):

        self.patch_type = self.config["model"]["patch"]["type"] if "type" in self.config["model"]["patch"] else "cart"
        self.patch_dist = self.config["model"]["patch"]["dist"] if "dist" in self.config["model"]["patch"] else 1.0
        self.patch_size = self.config["model"]["patch"]["size"] if "size" in self.config["model"]["patch"] else [3,3]
        self.patch_tconst = self.config["model"]["patch"]["tconst"] if "tconst" in self.config["model"]["patch"] else False
        self.exclude_center = self.config["model"]["patch"]["exclude_center"] if "exclude_center" in self.config["model"]["patch"] else 5

        if self.patch_type == "cart":
            dx = 2.0 / self.config['fe_steps']
            dy = 2.0 / self.config['fe_steps']
            dt = 0  # patch_dist * 2.0 / self.config['fe_steps'] # ToDo: Adjust distance for t when adding t for kernel

            origin_coord = sample["coords"].detach().cpu().clone().numpy()
            coord_neighbors = kernel_coords.create_cartesian_kernel(origin_coord,
                                                                    kernel_size=self.patch_size,
                                                                    patch_dist=self.patch_dist,
                                                                    delta_dist=[dx,dy])
        elif self.patch_type == "radial_full":

            # for cartesian fully sampled
            # nspokes >= (pi/2) * nFE
            # phie = (2*pi) / nspokes =  pi / (nFE * ( pi/2)) = 2 / nFE     # only pi, because all spokes until there fill the other half as well
            # spoke_FS = int(np.ceil(np.pi * 0.5 * self.config['fe_steps']))
            delta_phi = 2.0 / self.config["fe_steps"]  # in radian
            delta_fe = 2.0 / self.config['fe_steps']

            origin_coord = sample["coords"].detach().cpu().clone().numpy()
            coord_neighbors = kernel_coords.create_radial_kernel(origin_coord,
                                                                 kernel_size=self.patch_size,
                                                                 patch_dist=self.patch_dist,
                                                                 delta_dist_rad=[delta_fe,delta_phi],
                                                                 half = False)
            dx = delta_fe  # for later filtering of valid coords

        elif self.patch_type == "radial_half":
            # for cartesian fully sampled
            # nspokes >= (pi/2) * nFE
            # phie = (2*pi) / nspokes =  2*pi / (nFE * ( pi/2)) = 4 / nFE
            # spoke_FS = int(np.ceil(np.pi * 0.5 * self.config['fe_steps']))
            delta_phi = 2.0 / self.config["fe_steps"]  # in radian
            delta_fe = 2.0 / self.config['fe_steps']

            origin_coord = sample["coords"].detach().cpu().clone().numpy()
            coord_neighbors = kernel_coords.create_radial_kernel(origin_coord,
                                                                 kernel_size=self.patch_size,
                                                                 patch_dist=self.patch_dist,
                                                                 delta_dist_rad=[delta_fe,delta_phi],
                                                                 half = True)
            dx = delta_fe # for later filtering of valid coords

        elif self.patch_type == "radial_equi_full":
            # for cartesian fully sampled
            # nspokes >= (pi/2) * nFE
            # phie = (2*pi) / nspokes =  2*pi / (nFE * ( pi/2)) = 4 / nFE
            # spoke_FS = int(np.ceil(np.pi * 0.5 * self.config['fe_steps']))
            delta_phi = 2.0 / self.config["fe_steps"]  # in radian
            delta_fe = 2.0 / self.config['fe_steps']

            origin_coord = sample["coords"].detach().cpu().clone().numpy()
            coord_neighbors = kernel_coords.create_radial_equidistant_kernel(origin_coord,
                                                                 kernel_size=self.patch_size,
                                                                 patch_dist=self.patch_dist,
                                                                 delta_dist_rad=[delta_fe,delta_phi],
                                                                 half = False)
            dx = delta_fe # for later filtering of valid coords
        else:
            AssertionError("Patch Type unknown")

        ## Debug
        # import matplotlib.pyplot as plt
        # plt.scatter(coord_neighbors[[1], :, 1], coord_neighbors[[1], :, 2])
        # plt.scatter(origin_coord[[1], 1], origin_coord[[1], 2])
        # plt.show()

        # filter out desired coordinates (not at edge and not from center)
        edge_coords_idx = kernel_coords.get_edge_coords(origin_coord, coord_neighbors, dx)
        if self.exclude_center is not None:
            center_coords_idx = kernel_coords.get_center_coord(origin_coord, coord_neighbors, dx*self.exclude_center)
        else:
            center_coords_idx = np.zeros_like(edge_coords_idx)
        valid_coords_idx = np.logical_not(edge_coords_idx | center_coords_idx)

        sample["valid_coords_idx"] = valid_coords_idx
        sample["n_valid_coords"] = len(np.argwhere(valid_coords_idx == True))
        origin_coord = origin_coord[sample["valid_coords_idx"], :]
        coord_neighbors = coord_neighbors[sample["valid_coords_idx"], :, :]

        if self.patch_tconst:
            # set temporal value to same value for each subset
            origin_coord[:,0] = origin_coord[0,0]
            coord_neighbors[:,:,0] = coord_neighbors[:, : ,0]

        sample['coords_patch'] = torch.from_numpy(coord_neighbors).to(self.device)
        sample['coords_target'] = torch.from_numpy(origin_coord).to(self.device)
        # sample['coords_patch'] = sample['coords_patch'].to(self.device)

        return sample

    def post_process(self, output):
        """
        Convert the real output to a complex-valued output.
        The first half of the output is the real part, and the second half is the imaginary part.
        """
        output_complex = torch.complex(output[..., 0:self.config["out_dim"]].clone(),
                                       output[..., self.config["out_dim"]:].clone())
        return output_complex

    def train_batch(self, sample):

        self.network_kdata.train()
        torch.autograd.set_detect_anomaly(True)
        self.optimizer.zero_grad()
        self.optimizer_reg.zero_grad()

        sample['coords'], sample['targets'] = sample['coords'].to(self.device), sample['targets'].to(self.device)
        output = self.predict(sample['coords'])  ## only NIK

        with torch.no_grad():       ## sample patches
            sample = self.sample_patches(sample)
            coords_P, coords_T = sample["coords_patch"], sample['coords_target']

        if self.config["kreg"]["optim_type"] in ["joint", "joint_noBack", "joint_backP", "joint_backT"]:
            # Predict based on pre-defined backprop
            if self.config["kreg"]["optim_type"] == "joint":
                output_P, output_T = self.predict(coords_P), self.predict(coords_T)
            elif self.config["kreg"]["optim_type"] == "joint_backT":
                output_T = self.predict(coords_T)
                with torch.no_grad():
                    output_P = self.predict(coords_P)
            elif self.config["kreg"]["optim_type"] == "joint_backP":
                output_P = self.predict(coords_P)
                with torch.no_grad():
                    output_T = self.predict(coords_T)
            elif self.config["kreg"]["optim_type"] == "joint_noBack":
                with torch.no_grad():
                    output_P, output_T = self.predict(coords_P), self.predict(coords_T)

            loss_dc, reg = self.criterion(output, sample['targets'], sample['coords'])
            loss_reg, W_reg = self.criterion_reg(output_T, output_P, coords_T, coords_P)
            loss = loss_dc + self.criterion_reg.reg_lamda * loss_reg
            loss.backward()
            self.optimizer.step()

        elif self.config["kreg"]["optim_type"] in ["noreg", "onlyreg"]:
            loss_dc, reg = self.criterion(output, sample['targets'], sample['coords'])
            if self.config["kreg"]["optim_type"] == "onlyreg":
                output_P, output_T = self.predict(coords_P), self.predict(coords_T)
                loss_reg, W_reg = self.criterion_reg(output_T, output_P, coords_T, coords_P)
                loss = loss_reg
            else:
                with torch.no_grad():
                    output_P, output_T = self.predict(coords_P), self.predict(coords_T)
                    loss_reg, W_reg = self.criterion_reg(output_T, output_P, coords_T, coords_P)
                loss = loss_dc
            loss.backward()
            self.optimizer.step()

        elif self.config["kreg"]["optim_type"] in ["separate", "separate_backP", "separate_backT"]:
            loss_dc, reg = self.criterion(output, sample['targets'], sample['coords'])
            loss_dc.backward()
            self.optimizer.step()

            if self.config["kreg"]["optim_type"] == "separate":
                output_P, output_T = self.predict(coords_P), self.predict(coords_T)
            elif self.config["kreg"]["optim_type"] == "separate_backT":
                output_T = self.predict(coords_T)
                with torch.no_grad():
                    output_P = self.predict(coords_P)
            elif self.config["kreg"]["optim_type"] == "separate_backP":
                output_P = self.predict(coords_P)
                with torch.no_grad():
                    output_T = self.predict(coords_T)

            loss_reg, W_reg = self.criterion_reg(output_T, output_P, coords_T, coords_P)
            loss_reg_weighted = self.criterion_reg.reg_lamda * loss_reg
            loss_reg_weighted.backward()
            self.optimizer_reg.step()
            loss = loss_reg_weighted + loss_dc
        else:
            AssertionError("{} is no valid training strategy".format(self.config["kreg"]["optim_type"]))

        return loss, [loss_dc, loss_reg], W_reg

    def predict(self, coords):
        if coords.ndim == 3:
            nsamples = coords.shape[0]
            coords_flat = coords.reshape(-1, self.config["coord_dim"])
            features = self.pre_process(coords_flat)
        else:
            features = self.pre_process(coords)
        output = self.forward(features)
        output = self.post_process(output)
        if coords.ndim == 3:
            output = output.reshape(nsamples, -1, output.shape[-1])
        return output

    def test_batch(self, input=None, input_dim=None):
        """
        Test the network with a cartesian grid.
        if sample is not None, it will return image combined with coil sensitivity.
        """
        self.network_kdata.eval()
        self.conv_patch.eval() if hasattr(self, 'conv_patch') else None

        with torch.no_grad():
            nc = self.config['nc']  # len(self.config['coil_select'])  # nc = self.config['nc']
            if input is None:
                if input_dim is not None:  # if dim given, use these - otherwise default from config
                    assert len(input_dim) == self.config["coord_dim"]
                    nnav, nx, ny = input_dim
                else:
                    nx = self.config['nx']
                    ny = self.config['ny']
                    nnav = self.config['nnav']

                nav_min = float(self.config["dataset"]["nav_min"]) if "nav_min" in self.config["dataset"] else -1.
                nav_max = float(self.config["dataset"]["nav_max"]) if "nav_max" in self.config["dataset"] else -1.
                delta_nav = (nav_max - nav_min) / nnav

                # coordinates: kx, ky, nav
                kxs = torch.linspace(-1, 1 - 2 / nx, nx)
                kys = torch.linspace(-1, 1 - 2 / ny, ny)
                knav = torch.linspace(nav_min + delta_nav / nnav, nav_max - delta_nav / nnav,
                                  nnav) if nnav > 1 else torch.tensor(nav_min)

                grid_coords = torch.stack(torch.meshgrid(knav, kys, kxs, indexing='ij'), -1)  # nav, nx, ny,3
                dist_to_center = torch.sqrt(grid_coords[..., 1] ** 2 + grid_coords[..., 2] ** 2)
                dist_to_center = dist_to_center.unsqueeze(1).expand(-1, nc, -1, -1)  # nt, nc, nx, ny

                nDim = grid_coords.shape[-1]
                contr_split = 1

            else:
                grid_coords = input
                # grid_coords = grid_coords.to(self.device)
                nDim = grid_coords.shape[-1]
                contr_split = 1

            contr_split_num = np.ceil(grid_coords.shape[0] / contr_split).astype(int) # split t for memory saving
            kpred_list = []
            for t_batch in range(contr_split_num):
                grid_coords_batch = grid_coords[t_batch * contr_split:(t_batch + 1) * contr_split]
                grid_coords_batch = grid_coords_batch.reshape(-1, nDim).requires_grad_(False)
                # get prediction
                sample = {'coords': grid_coords_batch}
                features = self.pre_process(sample["coords"])   # encode time differently?
                kpred = self.forward(features)
                kpred = self.post_process(kpred)
                kpred_list.append(kpred)
            kpred = torch.concat(kpred_list, 0)

            if input is None or input_dim is not None:  # reshape if default or input_dim given (but not if coords)
                # TODO: clearning this part of code
                kpred = kpred.reshape(nnav, ny, nx, nc).permute(0,3,1,2) # nt, nc, ny, nx
                k_outer = 1 #
                kpred[dist_to_center >= k_outer] = 0

            if hasattr(self, "coil_factors") and self.coil_factors is not None:
                # rescale coil based on maximum intensity
                kpred = kpred * torch.tensor(self.coil_factors.reshape(1,-1,1,1)).to(kpred)

            return kpred

    def forward(self, input_coords):
        x = self.network_kdata(input_coords)
        return x


"""
The following code is a demo of mlp with sine activation function.
We suggest to only use the mlp model class to do the very specific 
mlp task: takes a feature vector and outputs a vector. The encoding 
and post-process of the input coordinates and output should be done 
outside of the mlp model (e.g. in the prepocess and postprocess 
function in your NIK model class).
"""

class Siren(nn.Module):
    def __init__(self, coord_dim, out_dim, hidden_features, num_layers, omega_0=30, exp_out=True) -> None:
        super().__init__()

        self.coord_dim = coord_dim
        self.hidden_features = hidden_features


        self.net = [SineLayer(hidden_features, hidden_features, is_first=True, omega_0=omega_0)]
        for i in range(num_layers - 1):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0))
        final_linear = nn.Linear(hidden_features, out_dim * 2)
        with torch.no_grad():
            # Initialize the weights without in-place operation
            final_linear.weight.data.uniform_(-np.sqrt(6 / hidden_features) / omega_0,
                                              np.sqrt(6 / hidden_features) / omega_0)
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

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
        x = self.net(features)
        return x


class SineLayer(nn.Module):
    """Linear layer with sine activation. Adapted from Siren repo"""

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
