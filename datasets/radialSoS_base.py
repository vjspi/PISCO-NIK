import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
from utils.mri import downsample_csm, scale_traj


class RadialSoSDatasetBase(object):
    def __init__(self, config):
        """
        Custom PyTorch Dataset for handling radial SoS MRI data.

        Parameters:
            config (dict): Configuration dictionary containing necessary parameters.
        """

        self.config = config
        self.config["dataset"]["nav_drift"] = self.config["dataset"]["nav_drift"] if "nav_drift" in self.config["dataset"] else False

        ## load data
        np_file_name = config['filename'] if "filename" in config else "test_reconSoS.npz"
        file_name = f"{config['data_root']}S{config['subject_name']}/{np_file_name}"
        data = np.load(file_name)

        ## kspace
        self.kdata = data["kspace"]             # expected input [ech, c, z, nFE*nPE]
        # trajectory
        self.traj = data["traj"]                # expected input [ech, nFE*nPE, 2]

        # Navigator
        if "nav_gt" in data:
            self.self_nav = data["nav_gt"]
            if self.self_nav.ndim == 2:
                self.self_nav = self.self_nav[:,config["slice"]]   # expected input [nPE, z]
        elif "pcaCurve" in data:
            self.self_nav = data["pcaCurve"]                    # expected input [nPE]
        else:
            AssertionError("No navigator signal")

        # self.weights = data["weights"] if "weights" in data.keys() else None         # expected input [ech, 1, nFE*nPE]

        # Reference
        try:
            self.ref = data["ref"]    # expected input [x, y, slices, ms, ech*dyn]
        except:
            self.ref = None

        # Sensmaps
        if "smaps" in data:
            self.csm = data["smaps"][...]               # load all slices sensitivity map,
        else:
            self.csm = data["csm"][...]     # expected input [c,x,y,z]

        # Dimensions
        self.im_size = self.csm.shape
        self.n_fe = self.config["fe_steps"]
        self.n_slices = self.config["n_slices"]
        self.n_coils = self.im_size[0]
        self.n_echo = self.kdata.shape[0]

        ## reshape data
        self.kdata = self.kdata.reshape(self.n_echo, self.n_coils, self.n_slices, -1, self.n_fe)  # (ech, coils, slices, spokes, FE)
        self.traj = self.traj.reshape(self.n_echo, -1, self.n_fe, 2)
        print("Trajectory shape:", self.traj.shape)
        self.n_spokes = self.kdata.shape[-2]  # imsize is coils * x * y

        # Weights
        try:
            self.weights = data["weights"]    # expected input [x, y, slices, ms, ech*dyn]
            self.weights = self.weights.reshape(self.weights.shape[0], -1,
                                                self.n_fe) if self.weights is not None else None
        except:
            self.weights = None


        # # %% Data processing
        if "acc_factor" in self.config["dataset"]:
            self.retrospectiveAcceleration(acc_factor=self.config["dataset"]["acc_factor"])

        if "fe_crop" in self.config["dataset"] and self.config["dataset"]["fe_crop"] < 1:
            self.downsampleResolution(fe_crop=self.config["dataset"]["fe_crop"])

        ## Preprocess navigator
        if self.config["dataset"]["nav_drift"]:
            self.navigator_removedrift(plot=False)
        self.nav_min = self.config["dataset"]["nav_min"] if "nav_min" in self.config["dataset"] else -1.
        self.nav_max = self.config["dataset"]["nav_max"] if "nav_max" in self.config["dataset"] else 1.
        self.scaleNavigator()
        self.traj = scale_traj(self.traj, max_value=1.0)

        #debug
        # plt.figure(figsize=(15, 5))
        # plt.plot(self.self_nav)
        # plt.title("Navigator signal")
        # plt.show()
        # plt.close()

        print("Original image shape", self.im_size)
        if self.n_coils != len(config["coil_select"]):
            print("Careful: coil selection does not match sensitivity maps")


        assert self.kdata.shape[-2] == self.traj.shape[-3]

    def navigator_removedrift(self, plot=True):
        nav_orig = self.self_nav.copy()
        coefficients = np.polyfit(np.arange(len(nav_orig)), nav_orig, 1)
        fitted_curve = np.polyval(coefficients, np.arange(len(nav_orig)))
        self.self_nav = self.self_nav - fitted_curve
        if plot:
            plt.figure(figsize=(10,3))
            plt.plot(nav_orig, '-g', label="Original navigator")
            plt.plot(self.self_nav, 'k',label="Shifted navigator")
            plt.plot(fitted_curve, 'r')
            plt.legend()
            plt.title("Navigator signal")
            plt.show()


    def scaleNavigator(self):
        # scale navigator signal
        min_val = self.nav_min
        max_val = self.nav_max
        original_min = np.min(self.self_nav)
        original_max = np.max(self.self_nav)
        if original_max != original_min:
            scaled_arr = min_val + (self.self_nav - original_min) * (max_val - min_val) / (original_max - original_min)
            self.self_nav = scaled_arr
        else:
            print("Assuming breath-hold data set since only one value for navigator signal")

    def retrospectiveAcceleration(self, acc_factor):
        nspoke_acc = int(self.kdata.shape[-2] / acc_factor)
        self.kdata = self.kdata[:, :, :, :nspoke_acc, :]
        self.traj = self.traj[:, :nspoke_acc, :, :]
        self.weights = self.weights[:, :nspoke_acc, :] if self.weights is not None else None
        self.self_nav = self.self_nav[:nspoke_acc]
        self.acc = acc_factor # ToDo: MArk all preprocessing
        self.n_spokes = self.kdata.shape[-2] # overwrite spokes

    def downsampleResolution(self, fe_crop):
        n_fe_crop = int(self.n_fe * fe_crop)
        crop_start = int((self.n_fe - n_fe_crop) / 2)
        self.traj = self.traj[:, :, crop_start:crop_start + n_fe_crop, :]
        self.weights = self.weights[:, :, crop_start:crop_start + n_fe_crop] if self.weights is not None else None
        self.kdata = self.kdata[..., crop_start:crop_start + n_fe_crop]
        assert self.kdata.shape[-1] == n_fe_crop
        assert np.all(np.sqrt(self.traj[..., 0] ** 2 + self.traj[..., 1] ** 2) < self.config["dataset"]["fe_crop"])
        self.traj *= (1 / self.config["dataset"]["fe_crop"])
        # Downsample coil sensitivity maps
        self.csm = downsample_csm(self.csm, self.config["dataset"]["fe_crop"])
        self.im_size = self.csm.shape
        self.n_fe = n_fe_crop  # overwrite nFE
        print("Downsampled image shape", self.im_size)
        print(
            "Warning: Trajectory FE direction got cropped and rescaled again - consider in final reconstruction voxel size ")

    def plotData(self, slice, echo=0):
        import medutils.visualization as vis
        import matplotlib.pyplot as plt
        vis.imshow(vis.plot_array(self.csm[:,:,:,slice]), title="Coil maps - slice {}".format(slice))
        vis.kshow(vis.plot_array(self.kdata[echo,:, slice,:,:]), title="Coil maps - slice {}".format(slice))
        plt.show()
    #
    # def scaleNavigator(self):
    #     # scale navigator signal
    #     if self.config["dataset"]["navigator_min"] == -1:
    #         self.self_nav = ((self.self_nav - (self.self_nav.max() + self.self_nav.min()) / 2) / (
    #                     self.self_nav.max() - self.self_nav.min())) * 2  # normalize to -1 to 1
    #     elif self.config["dataset"]["navigator_min"] == 0:
    #         self.self_nav = (self.self_nav - self.self_nav.min()) / (
    #                     self.self_nav.max() - self.self_nav.min())  # normalize to 0 to 1

        # # %% Data prepocessing
        # ### Acceleration (Phase encoding)
        # if "acc_factor" in self.config["dataset"]:
        #     nspoke_acc = int(kdata.shape[-2] / self.config["dataset"]["acc_factor"])
        #     kdata = kdata[:, :, :, :nspoke_acc, :]
        #     traj = traj[:, :nspoke_acc, :, :]
        #     self_nav = self_nav[:nspoke_acc]


        # ### Reduce resolution by cropping the frequency range (CAREFUL: Still rescaled back for training!)
        # if "fe_crop" in self.config["dataset"] and self.config["dataset"]["fe_crop"] < 1:
        #     n_fe_crop = int(n_fe * self.config["dataset"]["fe_crop"])
        #     crop_start = int((n_fe - n_fe_crop)/2)
        #     traj = traj[:,:,crop_start:crop_start+n_fe_crop,:]
        #     kdata = kdata[...,crop_start:crop_start+n_fe_crop]
        #     assert kdata.shape[-1] == n_fe_crop
        #     assert np.all(np.sqrt(traj[..., 0] ** 2 + traj[..., 1] ** 2) < self.config["dataset"]["fe_crop"])
        #     n_fe = n_fe_crop
        #     traj *= (1/self.config["dataset"]["fe_crop"])
        #     # Downsample coil sensitivity maps
        #     csm = downsample_csm(csm, self.config["dataset"]["fe_crop"])
        #     print("Downsampled image shape", im_size)
        #     print("Warning: Trajectory FE direction got cropped and rescaled again - consider in final reconstruction voxel size ")


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

    # def __len__(self):
    #     """
    #     Return the number of samples in the dataset.
    #
    #     Returns:
    #         int: The number of samples in the dataset.
    #     """
    #     return self.n_kpoints
    #
    # def __getitem__(self, index):
    #     """
    #     Get a sample from the dataset at the specified index.
    #
    #     Parameters:
    #         index (int): Index of the sample to retrieve.
    #
    #     Returns:
    #         dict: A dictionary containing 'coords', 'latent', 'targets', and 'kspoke' tensors for the sample.
    #     """
    #
    #     # In case the pool of samples is reduced (i.e. increment <1): Map the index to another the reduced range
    #     # mapped_index = index projected to desired increment region (e.g. with desired increment of 0.4 and 100 kpoints, index 60 corresponds to 20)
    #     no_samples = np.int(np.floor(self.increment * self.n_kpoints))
    #     index = index % no_samples      # maps all data points to specified range
    #
    #     # point wise sampling
    #     sample = {
    #         'coords': self.kcoords_flat[index],
    #         'latent': self.klatent_flat[index],
    #         'targets': self.kdata_flat[index],
    #         'kspoke': self.kspoke_flat[index]
    #     }
    #     return sample
