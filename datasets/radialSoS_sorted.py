from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
from utils.mri import downsample_csm, scale_traj
from datasets.radialSoS_base import RadialSoSDatasetBase


class RadialSoSDataset(RadialSoSDatasetBase, Dataset):
    def __init__(self, config):
        """
        Custom PyTorch Dataset for handling radial SoS MRI data.

        Parameters:
            config (dict): Configuration dictionary containing necessary parameters.
        """
        super().__init__(config)
        # self.config = config
        self.traj = scale_traj(self.traj, max_value=1.0)
        ### Crop region for calibration
        # if "calib_crop" in self.config and self.config["calib_crop"] < 1:
        #     n_fe_crop = int(n_fe * self.config["calib_crop"])
        #     crop_start = int((n_fe - n_fe_crop)/2)
        #     traj = traj[:,:,crop_start:crop_start+n_fe_crop,:]
        #     kdata = kdata[...,crop_start:crop_start+n_fe_crop]
        #     assert kdata.shape[-1] == n_fe_crop
        #     assert np.all(np.sqrt(traj[..., 0] ** 2 + traj[..., 1] ** 2) < self.config["calib_crop"])
        #     n_fe = n_fe_crop
        #     # traj *= (1/self.config["fe_crop"])
        #     print("Warning: Trajectory FE direction got cropped but NOT rescaled")

        ### crop data to desired echo/slice#

        # select data
        slice = self.config['slice']
        echo = self.config['echo'] if "echo" in self.config else 0
        self.n_echo = 1

        self.kdata = self.kdata[echo, :, slice, :, :]  # minimize data
        self.traj = self.traj[echo, :]
        self.weights = self.weights[echo, :, :] if self.weights is not None else None

        self.ref_all = self.ref
        self.ref = self.ref[:,:,[slice],:,:] if self.ref is not None else None
        self.ref = self.ref[:,:,:,:,[echo]] if self.ref is not None else None

        # normalize kdata
        self.kdata /= np.max(np.abs(self.kdata)) # ToDO: move normalization, consider coils
        if "scale_coils" in self.config["dataset"] and self.config["dataset"]["scale_coils"]:
            self.coil_factors = np.max(np.abs(self.kdata), axis=(1, 2))
            self.kdata /= self.coil_factors[:,None,None]
        self.create_sample_points()
        # initialize increment
        self.increment = 1  # 100 percent as default (all points are sampled)


    # def scaleNavigator(self):
    #     # scale navigator signal
    #     min_val = self.nav_min
    #     max_val = self.nav_max
    #     original_min = np.min(self.self_nav)
    #     original_max = np.max(self.self_nav)
    #     scaled_arr = min_val + (self.self_nav - original_min) * (max_val - min_val) / (original_max - original_min)
    #     self.self_nav = scaled_arr

        # if self.config["dataset"]["navigator_min"] == -1:
        #     self.self_nav = ((self.self_nav - (self.self_nav.max() + self.self_nav.min()) / 2) / (self.self_nav.max() - self.self_nav.min())) * 2     # normalize to -1 to 1
        # elif self.config["dataset"]["navigator_min"] == 0:
        #     self.self_nav = (self.self_nav - self.self_nav.min()) / (self.self_nav.max() - self.self_nav.min())                                  # normalize to 0 to 1

    def create_sample_points(self):
        ### move coil dimension to output
        nDim = self.config["coord_dim"]
        kcoords = np.zeros((self.n_spokes, self.n_fe, nDim))  # (nav, spokes, FE, 3) -> nav, ky, (contr, slices, spokes, FE, coils, 5) -> contr, kx, ky, nc, nav
        klatent = np.zeros((self.n_spokes, self.n_fe, 1))
        # kc = torch.linspace(-1, 1, n_coils)
        ## Save spoke number for each sample
        kspoke = np.zeros((self.n_spokes, self.n_fe, 1))
        # contr = torch.linspace(-1, 1, n_contr)

        # kcoords[:, :, :, :, 0] = np.reshape(self_nav, (1, n_spokes, n_fe))
        # kcoords[..., 0] = np.reshape(kc, (n_coils, 1, 1))
        kcoords[..., 1] = np.reshape(self.traj[..., 0], (1, self.n_spokes, self.n_fe))  # ky
        kcoords[..., 2] = np.reshape(self.traj[..., 1], (1, self.n_spokes, self.n_fe))  # kx
        kcoords[..., 0] = np.reshape(self.self_nav, (1, self.n_spokes, 1))
        klatent[..., 0] = np.reshape(self.self_nav, (1, self.n_spokes, 1))
        # kcoords[:, :, :, 3] = np.reshape(self_nav, (1, n_spokes, n_fe))
        kspoke[..., 0] = np.reshape(np.linspace(0, 1, self.n_spokes), (self.n_spokes, 1))

        ### put coils to output
        assert self.kdata.shape[0] == self.n_coils        # coils x spokes x FE
        kdata = np.transpose(self.kdata, (1,2,0))    # spokes x FE x coils

        # sort data from center to outer edge if calib region is required
        # ToDo: Clean condition
        if "patch_schedule" in self.config and self.config["patch_schedule"]["calib_region"] < 1:
            kcoords = np.reshape(kcoords.astype(np.float32), (-1, nDim))
            klatent = np.reshape(klatent.astype(np.float32), (-1, 1))
            kdatapoints = np.reshape(kdata.astype(np.complex64), (-1, self.n_coils))
            kspoke = np.reshape(kspoke.astype(np.float32), (-1, 1))

            dist_to_center = np.sqrt(kcoords[..., 1] ** 2 + kcoords[..., 2] ** 2)
            idx = np.argsort(dist_to_center)

            self.kcoords = kcoords[idx].astype(np.float32)
            self.klatent = klatent[idx].astype(np.float32)
            self.kdatapoints = kdatapoints[idx].astype(np.complex64)  # (nav*spokes*FE, 1)
            self.kspoke = kspoke[idx].astype(np.float32)

        else:
            self.kcoords = np.reshape(kcoords.astype(np.float32), (-1, nDim))
            self.klatent = np.reshape(klatent.astype(np.float32), (-1, 1))
            self.kdatapoints = np.reshape(kdata.astype(np.complex64), (-1, self.n_coils))  # (nav*spokes*FE, 1)
            self.kspoke = np.reshape(kspoke.astype(np.float32), (-1, 1))


        self.kcoords_flat = torch.from_numpy(self.kcoords)
        self.klatent_flat = torch.from_numpy(self.klatent)
        self.kdata_flat = torch.from_numpy(self.kdatapoints)
        self.kspoke_flat = torch.from_numpy(self.kspoke)

        self.n_kpoints = self.kcoords.shape[0]


    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.n_kpoints

    def __getitem__(self, index):
        """
        Get a sample from the dataset at the specified index.

        Parameters:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing 'coords', 'latent', 'targets', and 'kspoke' tensors for the sample.
        """
        
        # In case the pool of samples is reduced (i.e. increment <1): Map the index to another the reduced range
        # mapped_index = index projected to desired increment region (e.g. with desired increment of 0.4 and 100 kpoints, index 60 corresponds to 20)
        no_samples = int(np.floor(self.increment * self.n_kpoints))
        index = index % no_samples      # maps all data points to specified range

        # point wise sampling
        sample = {
            'coords': self.kcoords_flat[index],
            'latent': self.klatent_flat[index],
            'targets': self.kdata_flat[index],
            'kspoke': self.kspoke_flat[index]
        }
        return sample
