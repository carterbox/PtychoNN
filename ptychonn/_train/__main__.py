import glob
import logging
import os
import pathlib

import click
import numpy as np
import numpy.typing as npt
import torch
import torchinfo

import ptychonn.model
import ptychonn.plot

logger = logging.getLogger(__name__)


@click.command(name='train')
@click.argument(
    'data_dir',
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    'out_dir',
    type=click.Path(
        exists=False,
        writable=True,
        dir_okay=True,
        file_okay=False,
        path_type=pathlib.Path,
    ),
)
def train_cli(
    data_dir: pathlib.Path,
    out_dir: pathlib.Path,
):
    """Train a model from diffraction patterns and reconstructed patches.

    Expects a `DATA_DIR` of NPZ files. Each contains two parameters:
    `reciprocal` and `real` which are the diffraction patterns and
    reconstructed (complex-valued) patches. The shapes of these two arrays
    match. The data from every file in the folder will be concatinated into one
    training/validation set. All the training artifacts are saved into
    `OUT_DIR`.
    """
    logging.basicConfig(level=logging.INFO, )

    dataslist = []
    patchlist = []

    for name in glob.glob(str(data_dir / '*.npz')):
        print(name)
        with np.load(name) as f:
            dataslist.append(f['reciprocal'])
            patchlist.append(f['real'])

    data = np.concatenate(dataslist, axis=0)
    patches = np.concatenate(patchlist, axis=0)

    # NOTE: @carterbox AFAIK, the only preprocessing of the training data is a
    # centering of the phase in the center 3rd of the reconstructed patches.
    # The diffraction patterns are converted to float32 and otherwise
    # unaltered.
    patches = np.angle(patches).astype('float32')
    patches -= np.mean(
        patches[..., patches.shape[-2] // 3:-patches.shape[-2] // 3,
                patches.shape[-1] // 3:-patches.shape[-1] // 3], )

    os.makedirs(out_dir, exist_ok=True)

    train(
        X_train=data,
        Y_train=patches,
        out_dir=out_dir,
        epochs=128,
        batch_size=64,
        epochs_per_half_cycle=16,
    )


def train(
    X_train: npt.NDArray[np.float],
    Y_train: npt.NDArray[np.float],
    out_dir: pathlib.Path,
    load_model_path: pathlib.Path | None = None,
    epochs: int = 1,
    batch_size: int = 64,
    epochs_per_half_cycle: int = 6,
):
    """Train a PtychoNN model.

    Parameters
    ----------
    X_train (N, WIDTH, HEIGHT)
        The diffraction patterns.
    Y_train (N, WIDTH, HEIGHT)
        The corresponding reconstructed patches for the diffraction patterns.
    out_dir
        A folder where all the training artifacts are saved.
    load_model_path
        Load a previous model's parameters from this file.
    """
    assert Y_train.dtype == np.float32
    assert np.all(np.isfinite(Y_train))
    assert X_train.dtype == np.float32
    assert np.all(np.isfinite(X_train))
    logger.info("Creating the training model...")

    trainer = Trainer(
        model=ptychonn.model.ReconSmallPhaseModel(
            nconv=X_train.shape[-1] // 2,
        ),
        batch_size=batch_size * torch.cuda.device_count(),
        output_path=out_dir,
        output_suffix='',
    )
    trainer.setTrainingData(
        X_train,
        Y_train,
        valid_data_ratio=0.1,
    )
    trainer.setOptimizationParams(
        epochs_per_half_cycle=epochs_per_half_cycle,
        max_lr=1e-3,
        min_lr=1e-4,
    )
    trainer.initModel(model_params_path=load_model_path, )
    trainer.run(epochs)

    trainer.plotLearningRate(
        save_fname=out_dir / 'learning_rate.svg',
        show_fig=False,
    )
    ptychonn.plot.plot_metrics(
        trainer.metrics,
        save_fname=out_dir / 'metrics.svg',
        show_fig=False,
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
    return loss.sum()


class Trainer():
    """A object that manages training PtychoNN
    """
    def __init__(
        self,
        model: ptychonn.model.ReconSmallPhaseModel,
        batch_size: int,
        output_path: pathlib.Path,
        output_suffix: str,
    ):
        logger.info("Initializing the training procedure...")
        self.model = model
        self.batch_size = batch_size
        self.output_path = output_path
        self.output_suffix = output_suffix
        self.epoch = 0

    def setTrainingData(
        self,
        X_train_full: np.ndarray,
        Y_ph_train_full: np.ndarray,
        valid_data_ratio: float = 0.1,
    ):
        logger.info("Setting training data...")

        self.H, self.W = X_train_full.shape[-2:]

        self.X_train_full = torch.tensor(
            X_train_full[:, None, ...],
            dtype=torch.float32,
        )
        self.Y_ph_train_full = torch.tensor(
            Y_ph_train_full[:, None, ...],
            dtype=torch.float32,
        )
        self.ntrain_full = self.X_train_full.shape[0]

        self.valid_data_ratio = valid_data_ratio
        self.nvalid = int(self.ntrain_full * self.valid_data_ratio)
        self.ntrain = self.ntrain_full - self.nvalid

        self.train_data_full = torch.utils.data.TensorDataset(
            self.X_train_full,
            self.Y_ph_train_full,
        )

        self.train_data, self.valid_data = torch.utils.data.random_split(
            self.train_data_full,
            [self.ntrain, self.nvalid],
        )
        self.trainloader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=False,
        )

        self.validloader = torch.utils.data.DataLoader(
            self.valid_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
        )

        self.iters_per_epoch = self.ntrain // self.batch_size + (
            self.ntrain % self.batch_size > 0)

    def setOptimizationParams(
        self,
        epochs_per_half_cycle: int = 6,
        max_lr: float = 5e-4,
        min_lr: float = 1e-4,
    ):
        logger.info("Setting optimization parameters...")

        # TODO: Move this note about iterations into the documentation string
        # after figuring out what it means. Paper recommends 2-10 number of
        # iterations
        self.epochs_per_half_cycle = epochs_per_half_cycle
        self.iters_per_half_cycle = epochs_per_half_cycle * self.iters_per_epoch

        logger.info(
            "LR step size is: %d which is every %d epochs",
            self.iters_per_half_cycle,
            self.iters_per_half_cycle / self.iters_per_epoch,
        )

        self.max_lr = max_lr
        self.min_lr = min_lr

        self.certainty = torch.ones(
            *self.Y_ph_train_full.shape[-2:],
            requires_grad=True,
        )
        self.criterion = gaussian_nll_loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.max_lr,
        )
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            max_lr=self.max_lr,
            base_lr=self.min_lr,
            step_size_up=self.iters_per_half_cycle,
            cycle_momentum=False,
            mode='triangular2',
        )

    def initModel(self, model_params_path: pathlib.Path | None = None):
        """Load parameters from the disk then model to the GPU(s)."""

        self.model_params_path = model_params_path
        if model_params_path is not None:
            self.model.load_state_dict(torch.load(self.model_params_path))
        torchinfo.summary(self.model, (1, 1, self.H, self.W), device="cpu")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            logger.info("Let's use %d GPUs!", torch.cuda.device_count())
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = torch.nn.DataParallel(
                self.model,
                device_ids=None,  # Default all devices
            )

        self.certainty = self.certainty.to(self.device)
        self.model = self.model.to(self.device)

        self.scaler = torch.cuda.amp.GradScaler()

        logger.info("Setting up metrics...")
        self.metrics = {
            'losses': [],
            'val_losses': [],
            'lrs': [],
            'best_val_loss': np.inf
        }
        logger.info(self.metrics)

    def train(self):
        tot_loss = 0.0
        loss_ph = 0.0

        for (ft_images, phs) in self.trainloader:

            # Move everything to device
            ft_images = ft_images.to(self.device)
            phs = phs.to(self.device)

            # Divide cumulative loss by number of batches-- slightly inaccurate
            # because last batch is different size
            pred_phs = self.model(ft_images)
            loss_p = self.criterion(pred_phs, phs, self.certainty)
            # Monitor phase loss but only within support (which may not be same
            # as true amp)
            loss = loss_p
            # Use equiweighted amps and phase

            # Zero current grads and do backprop
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            batch_fraction = len(ft_images) / len(self.train_data)
            tot_loss += loss.detach().item() * batch_fraction
            loss_ph += loss_p.detach().item() * batch_fraction

            # Update the LR according to the schedule -- CyclicLR updates each
            # batch
            self.scheduler.step()
            self.metrics['lrs'].append(self.scheduler.get_last_lr())
            self.scaler.update()

        self.metrics['losses'].append([tot_loss, loss_ph])

    def validate(self, epoch: int):
        tot_val_loss = 0.0
        val_loss_ph = 0.0
        for (ft_images, phs) in self.validloader:
            ft_images = ft_images.to(self.device)
            phs = phs.to(self.device)
            pred_phs = self.model(ft_images)

            val_loss_p = self.criterion(pred_phs, phs, self.certainty)
            val_loss = val_loss_p

            batch_fraction = len(ft_images) / len(self.valid_data)
            tot_val_loss += val_loss.detach().item() * batch_fraction
            val_loss_ph += val_loss_p.detach().item() * batch_fraction

        self.metrics['val_losses'].append([tot_val_loss, val_loss_ph])

        self.saveMetrics(self.metrics, self.output_path, self.output_suffix)

        # Update saved model if val loss is lower
        if (tot_val_loss < self.metrics['best_val_loss']):
            logger.info(
                "Saving improved model after Val Loss improved from %.3e to %.3e",
                self.metrics['best_val_loss'],
                tot_val_loss,
            )
            self.metrics['best_val_loss'] = tot_val_loss
            self.updateSavedModel(self.model, self.output_path,
                                  self.output_suffix)

            import tifffile
            os.makedirs(self.output_path / 'reference', exist_ok=True)
            os.makedirs(self.output_path / 'inference', exist_ok=True)
            if epoch <= 1:
                tifffile.imwrite(
                    self.output_path / f'reference/{epoch:05d}.tiff',
                    phs[0, 0].detach().cpu().numpy().astype(np.float32))
            tifffile.imwrite(
                self.output_path / f'inference/{epoch:05d}.tiff',
                pred_phs[0, 0].detach().cpu().numpy().astype(np.float32))

            ptychonn.plot.plot_metrics(
                self.metrics,
                save_fname=self.output_path / 'metrics.svg',
                show_fig=False,
            )

    # TODO: Use a callback instead of a static method for saving the model?

    @staticmethod
    def updateSavedModel(
        model: ptychonn.model.ReconSmallPhaseModel,
        path: pathlib.Path,
        output_suffix: str = '',
    ):
        """Update saved model if validation loss is minimum."""
        if not os.path.isdir(path):
            os.mkdir(path)
        fname = path / ('best_model' + output_suffix + '.pth')
        logger.info("Saving best model as %s", fname)
        torch.save(model.state_dict(), fname)

    @staticmethod
    def saveMetrics(
        metrics: dict,
        path: pathlib.Path,
        output_suffix: str = '',
    ):
        np.savez(path / ('metrics' + output_suffix + '.npz'), **metrics)

    def run(self, epochs: int, output_frequency: int = 1):
        """The main training loop"""

        for epoch in range(epochs):

            #Set model to train mode
            self.model.train()

            #Training loop
            self.train()

            #Switch model to eval mode
            self.model.eval()

            #Validation loop
            self.validate(epoch)

            if epoch % output_frequency == 0:
                logger.info(
                    'Epoch: %d | FT  | Train Loss: %1.03e | Val Loss: %1.03e',
                    epoch,
                    self.metrics['losses'][-1][0],
                    self.metrics['val_losses'][-1][0],
                )
                logger.info(
                    'Epoch: %d | Ph  | Train Loss: %1.03e | Val Loss: %1.03e',
                    epoch,
                    self.metrics['losses'][-1][1],
                    self.metrics['val_losses'][-1][1],
                )
                logger.info(
                    'Epoch: %d | Ending LR: %1.03e',
                    epoch,
                    self.metrics['lrs'][-1][0],
                )

    def plotLearningRate(
        self,
        save_fname: pathlib.Path | None = None,
        show_fig: bool = True,
    ):
        batches = np.linspace(
            0,
            len(self.metrics['lrs']),
            len(self.metrics['lrs']) + 1,
        )
        epoch_list = batches / self.iters_per_epoch

        import matplotlib.pyplot as plt

        f = plt.figure()
        plt.plot(epoch_list[1:], self.metrics['lrs'], 'C3-')
        plt.grid()
        plt.ylabel("Learning rate")
        plt.xlabel("Epoch")
        plt.tight_layout()

        if save_fname is not None:
            plt.savefig(save_fname)
        if show_fig:
            plt.show()
        else:
            plt.close(f)
