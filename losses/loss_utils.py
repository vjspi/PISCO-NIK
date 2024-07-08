
def l1_loss_from_difference(error):
    return torch.mean(torch.abs(error))
import random
import matplotlib.pyplot as plt
import torch

def l2_loss_from_difference(error):
    return torch.mean(torch.abs(error ** 2))

def huber_loss_from_difference(error, delta=1.0):
    quadratic_loss = 0.5 * (error ** 2)
    linear_loss = delta * (torch.abs(error) - 0.5 * delta)
    loss = torch.where(torch.abs(error) <= delta, quadratic_loss, linear_loss)
    return torch.mean(loss)

def loss_correlation(y1,y2):
    mean_orig = torch.mean(y1, dim=1)
    mean_fit = torch.mean(y2, dim=1)

    numerator = torch.sum(
        (y1 - mean_orig.unsqueeze(1))
        * (y2 - mean_fit.unsqueeze(1)),
        dim=1
    )
    denominator = torch.sqrt(
        torch.sum((y1 - mean_orig.unsqueeze(1)) ** 2, dim=1) *
        torch.sum((y2 - mean_fit.unsqueeze(1)) ** 2, dim=1)
    )
    return 1 - torch.mean(numerator / denominator)
def pdist_loss(X, Y, reduce_dim=1):
    """
    Compute the pdist_loss function between all combinations of rows in matrices X and Y.

    Parameters:
    X (torch.Tensor): First complex-valued matrix with shape (M, N).
    Y (torch.Tensor): Second complex-valued matrix with shape (P, N).

    Returns:
    torch.Tensor: Loss matrix with shape (M, P) where each element (i, j) corresponds to the losses between row i of X and row j of Y.
    """
    mag_input = torch.abs(X)
    mag_target = torch.abs(Y[:, None])

    cross = torch.abs(X.real * Y.imag[:, None] - X.imag * Y[:, None].real)
    angle = torch.atan2(X.imag, X.real) - torch.atan2(Y[:, None].imag, Y[:, None].real)
    ploss = torch.abs(cross) / (mag_input + 1e-8)

    aligned_mask = (torch.cos(angle) < 0).bool()
    final_term = torch.where(aligned_mask, mag_target + (mag_target - ploss), ploss)

    loss_matrix = final_term + torch.abs(mag_input - mag_target)  # + F.mse_loss(mag_input, mag_target)
    loss = torch.mean(loss_matrix, dim=reduce_dim)

    # debug
    plt.imshow(X.abs().detach().cpu().numpy())
    plt.ylabel("Subset"), plt.title("Weights")
    plt.show()
    plt.imshow(loss.detach().cpu().numpy())
    plt.ylabel("Subset"), plt.title("Mean distance to other subsets")
    plt.show()
    return loss
