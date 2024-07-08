import torch
from losses.loss_utils import *
class PiscoLoss(torch.nn.Module):

    def __init__(self, config, model):
        super().__init__()
        self.out_dim = config["out_dim"]
        self.coord_dim = config["coord_dim"]
        # self.batch_dim = config["batch_size"]
        self.kernel_size = config["model"]["patch"]["size"]

        assert "kreg" in config
        ## Optimization settings
        self.optim = config["kreg"]["optim"] if "optim" in config["kreg"] else "Adam"
        self.lr = float(config["kreg"]["lr"]) if "lr" in config["kreg"] else 3e-5
        self.mini_batch = int(config["kreg"]["mini_batch"]) if "mini_batch" in config["kreg"] else 0

        # Design choices for self-supervised losses calcuation
        self.loss_type = config["kreg"]["loss_type"] if "loss_type" in config["kreg"] else "shift_mean"
        self.loss_norm = config["kreg"]["loss_norm"] if "reg_alpha" in config["kreg"] else "L1_real_imag"

        # Hyperparameter for W solving
        self.overdet_factor = config["kreg"]["overdetermination"] if "overdetermination" in config["kreg"] else 1.0
        self.max_sets = config["kreg"]["max_sets"] if "max_sets" in config["kreg"] else 20
        self.reg_lamda = float(config["kreg"]["reg_lamda"]) if 'reg_lamda' in config["kreg"] else 0.01
        self.reg_type = config["kreg"]["reg_type"] if "reg_type" in config["kreg"] else None
        self.reg_alpha = float(config["kreg"]["reg_alpha"]) if "reg_alpha" in config["kreg"] else 0.00
        self.complex_handling = config["kreg"]["complex_handling"] if "complex_handling" in config["kreg"] else "mag_phase"
        self.sort_sets = config["kreg"]["sort_sets"] if "sort_sets" in config["kreg"] else None
        assert self.complex_handling == "mag_phase" or self.complex_handling == "img_real"

        # define out dimensions depending if coils are sampeld individually or all
        self.out_dim_patch = self.out_dim
        self.out_dim_target = self.out_dim if self.mini_batch == 0 else self.mini_batch
        if self.out_dim_target > 6:
            print("Selecting 6 coils randomly each iteration to allow enough subset combinations")
            self.out_dim_target = 6
            self.mini_batch = 6

        # number of possible linear combinations
        self.num_sets_func = lambda x: int(x // self.n_samples_per_set) if int(x // self.n_samples_per_set) < self.max_sets else self.max_sets

        print("Loss for k-REG used: {}".format(self.loss_type))
        print("Complex numbers split to: {}".format(self.complex_handling))
        print("k-Reg weight solving regularized by {} with weight {}".format(self.reg_type, self.reg_alpha))

    def forward(self, output, output_patch, coords, coords_patch, reduce=True):
        '''
        output_patch = [batch, n_neighbors, out_dim]
        output = [batch, out_dim]
        '''

        if self.mini_batch != 0:
            idx = random.sample(range(self.out_dim), self.mini_batch)
            output = output[..., idx]

        self.n_neighbors = output_patch.shape[1]
        self.n_weights = self.n_neighbors * self.out_dim_patch * self.out_dim_target     # (n_neighbors * out_dim, out_dim) - out_dim normally the number of coils
        self.n_samples_per_set = int(self.n_weights * self.overdet_factor)          ## number of samples to solve for kernel weights
        self.num_sets_func = lambda x: int(x // self.n_samples_per_set) if int(x // self.n_samples_per_set) < self.max_sets else self.max_sets

        # calculate the samples needed/used
        num_samples = output.shape[0]
        self.num_sets = self.num_sets_func(num_samples)
        num_used_samples = int(self.num_sets * self.n_samples_per_set)

        output = output.unsqueeze(-1)                                           # batch, out_dim_targ, 1
        output_patch = output_patch.reshape(num_samples, -1).unsqueeze(-1)      # batch, patch * out_dim_patch, 1
        coords_patch = coords_patch.reshape(num_samples, -1, self.coord_dim)    # batch, n_neighbors, coords_dim

        # Target and patch samples
        T = output[:num_used_samples, :, :]
        C_T = coords[:num_used_samples, :]
        C_P = coords_patch[:num_used_samples, :]
        P = output_patch[:num_used_samples, :, :]

        if self.sort_sets == "temp":
            _, temp_sort_idx = torch.sort(C_T[:,0])
            C_T = C_T[temp_sort_idx]
            C_P = C_P[temp_sort_idx]
            T = T[temp_sort_idx]
            P = P[temp_sort_idx]

        C_T = C_T.reshape(self.num_sets, self.n_samples_per_set, self.coord_dim)  # here num_patches = numweightsS, W, W - s set of W weight comcinations
        C_P = C_P.reshape(self.num_sets, self.n_samples_per_set, self.n_neighbors, self.coord_dim)  # here num_patches = numweightsS, W, W - s set of W weight comcinations
        # max_deltat = torch.max(C_T[:, -1, 0] - C_T[:, 0, 0]).item()
        # max_deltap = torch.max(C_P[:, -1, 0, 0] - C_P[:, 0, 0, 0]).item()
        # plot_coords_scatter(data=C_T.clone().detach().cpu(), data_patch=C_P.clone().detach().cpu())
        # plot_coords_scatter2D(data=C_T[...,1:].clone().detach().cpu(), data_patch=None)
        d_T = C_T[..., 1] ** 2 + C_T[..., 2] ** 2

        T = T.reshape(self.num_sets, self.n_samples_per_set, self.out_dim_target)
        P = P.reshape(self.num_sets, self.n_samples_per_set, self.n_neighbors*self.out_dim_patch) # here num_patches = numweightsS, W, W - s set of W weight comcinations

        # Debug
        # plot_PTD(P, T, d_T)

        ## to solve like Grappa we need
        #   P  : (npatch, nnxnc)
        #   T  : (npatch, nc)
        #   -> W = (nnxnc, nc)
        # PW = T
        # P^H P W = P^H T
        # W = ( P^H P)^-1 * PH T
        # T and P come as [nsets, npatch, nc] and [nsets, npatch, nnxnc]
        # PhP = P.conj().T @ P
        # PhT = P.conj().T @ T
        PhP = P.mH @ P   # [neighbors*out_dim, nsamples]* [nsamples, neighbors*out_dim]
        PhT = P.mH @ T   # [neighbors*out_dim, nsamples]* [nsamples, out_dim]

        if self.reg_type == "Tikhonov":
            lamda0 = self.reg_alpha * torch.linalg.matrix_norm(PhP) / (PhP.shape[-1])  # frobenius norm, calculated for each batch
            # lamda0 = self.reg_alpha * torch.ones(PhP.shape[0]).to(PhP) # / (PhP.shape[-1])  # same regularization for each batch
            I = torch.eye(PhP.shape[-1]).to(PhP)
            W = torch.linalg.solve((PhP + lamda0.view(-1,1,1) * I), PhT)                # add regularization to each batch individually
        else:
            W = torch.linalg.solve(PhP, PhT)

        # Debug # Todo
        # import medutils.visualization as vis
        # vis.imshow(vis.plot_array(torch.abs(W[0, ...].reshape(self.n_neighbors, self.out_dim_patch, self.out_dim_target)).detach().cpu().numpy()))
        # plt.show()

        # Calculate losses on solved weight sets
        W = W.reshape(self.num_sets, -1) # n_s, n_weights
        W_magphase = torch.stack([W.abs(), W.angle()], axis=-1)

        # Debug
        # plt.show()
        # plt.figure(figsize=(20, 5))
        # plt.plot(np.angle(W.reshape(self.num_sets, -1).T.detach().cpu().numpy()), alpha=0.5, color="b")
        # plt.plot(np.mean(np.angle(W.reshape(self.num_sets, -1).detach().cpu().numpy()), axis=0), alpha=1.0, linewidth=1,
        #          color="k")


        #### Distance losses ####
        if self.loss_type in ["L1_dist", "L1C_dist"]:
            W_stack = torch.concat([W.real, W.imag], axis=-1)
            W_error = torch.cdist(W_stack, W_stack, p=1) / W_stack.shape[-1]    # normalize by number of weights
            W_shift_magphase = torch.stack([W_error, torch.zeros_like(W_error)], axis=-1)
            W_m_magphase = W_shift_magphase
        else:
            AssertionError("Distance losses {} not supported".format(self.loss_type))


        #### Loss norm ####
        # ToDo: define error
        error = W_error
        if self.loss_norm in ["L1","L1_dist"]:
            error = l1_loss_from_difference(error)
        elif self.loss_norm in ["L2","L2_dist"]:
            error = l2_loss_from_difference(error)
        elif self.loss_norm in ["huber"]:
            error = huber_loss_from_difference(error)
        else:
            AssertionError("Loss norm {} not supported".format(self.loss_norm))

        if reduce:
            return torch.mean(error),  [W_m_magphase, W_shift_magphase, W_magphase]
        else:
            return error,  [W_m_magphase, W_shift_magphase, W_magphase]
