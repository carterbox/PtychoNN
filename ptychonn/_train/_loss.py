import torchmetrics
import torch
import numpy as np

betas = np.array([
    # Reduce the number of levels (betas) for small images
    # 0.0448,  # 128
    0.2856,  # 64
    0.3001,  # 32
    0.2363,  # 16
    0.1333,  # 8
])
betas /= np.sum(betas)
betas = tuple(betas)


def msssim_loss(preds, target):
    """A loss function that must be strictly positive to allow for plotting on log scale."""
    return 1.0 - torchmetrics.functional.multiscale_structural_similarity_index_measure(
        preds=preds[:, None],
        target=target[:, None],
        data_range=1.0,
        normalize='relu',
        betas=betas,
        sigma=0.5,
        kernel_size=5,
    )

def msssim_and_poisson_nll_loss(preds, target):
    """Combine metrics for structure and photon statistics."""
    return msssim_loss(preds, target) + torch.nn.functional.poisson_nll_loss(
        preds, target,
    )

def msssim_and_gaussian_nll_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    var: torch.Tensor,
    eps: float = 1e-6,):
    """Combine metrics for structure and photon statistics."""
    return msssim_loss(input, target) + gaussian_nll_loss(
        input,
        target,
        var,
        eps,
    )

def gaussian_nll_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    var: torch.Tensor,
    eps: float = 1e-6,
):
    """Plucked from torch.nn.functional; skips input shape checks."""
    if torch.overrides.has_torch_function_variadic(input, target, var):
        return torch.overrides.handle_torch_function(
            gaussian_nll_loss,
            (input, target, var),
            input,
            target,
            var,
            eps=eps,
        )

    # Entries of var must be non-negative
    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    # Calculate the loss
    loss = 0.5 * (torch.log(var) + (input - target)**2 / var)
    return loss.mean()

