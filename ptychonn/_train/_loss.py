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


def msssim_loss(preds, target, *args, **kwargs):
    """A loss function that must be strictly positive to allow for plotting on log scale."""
    assert torch.all(torch.isfinite(preds))
    assert torch.all(torch.isfinite(target))
    x = 1.0 - torchmetrics.functional.multiscale_structural_similarity_index_measure(
        preds=preds,
        target=target,
        data_range=np.pi*2,
        sigma=0.5,
        kernel_size=5,
        normalize='relu',
        betas=betas,
    )
    assert torch.isfinite(x)
    return x

def ssim_loss(preds, target, *args, **kwargs):
    """A loss function that must be strictly positive to allow for plotting on log scale."""
    assert torch.all(torch.isfinite(preds))
    assert torch.all(torch.isfinite(target))
    x = 1.0 - torchmetrics.functional.structural_similarity_index_measure(
        preds=preds,
        target=target,
        data_range=np.pi*2,
    )
    assert torch.isfinite(x)
    return x

def msssim_and_poisson_nll_loss(preds, target, *args, **kwargs):
    """Combine metrics for structure and photon statistics."""
    return msssim_loss(preds, target) + torch.nn.functional.poisson_nll_loss(
        preds, target, log_input=False,
    )

def msssim_and_gaussian_nll_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    var: torch.Tensor,
    eps: float = 1e-6,
):
    """Combine metrics for structure and photon statistics."""
    return msssim_loss(input, target) + gaussian_nll_loss(
        input,
        target,
        var,
        eps,
    )

def ssim_and_gaussian_nll_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    var: torch.Tensor,
    eps: float = 1e-6,
):
    """Combine metrics for structure and photon statistics."""
    return ssim_loss(input, target) + gaussian_nll_loss(
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

