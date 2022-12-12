import logging
import os
import pathlib

from torch.utils.data import TensorDataset, DataLoader
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
    'data_path',
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    'patch_path',
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
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
    data_path: pathlib.Path,
    patch_path: pathlib.Path,
    out_dir: pathlib.Path,
):
    """Train a model from diffraction patterns and reconstructed patches."""
    logging.basicConfig(level=logging.INFO, )
    data = np.load(data_path).astype('float32')
    patches = np.angle(np.load(patch_path)).astype('float32')
    patches -= np.mean(patches)
    data /= data.max()
    train(
        X_train=data,
        Y_train=patches,
        iteration_out_path=out_dir,
        epochs=100,
        batch_size=64,
    )


def train(
    X_train: npt.NDArray[np.float],
    Y_train: npt.NDArray[np.float],
    iteration_out_path: pathlib.Path,
    load_model_path: pathlib.Path | None = None,
    epochs: int = 1,
    batch_size: int = 64,
):
    """Train a PtychoNN model.

    Parameters
    ----------
    X_train (N, WIDTH, HEIGHT)
        The diffraction patterns.
    Y_train (N, WIDTH, HEIGHT)
        The corresponding reconstructed patches for the diffraction patterns.
    best_model_params_path
        Where you want to save the trained model. Will be overwritten.
    iteration_out_path
        ?
    load_model_path
        Load a previous model but don't overwrite.
    """
    assert Y_train.dtype == np.float32
    assert np.all(np.isfinite(Y_train))
    assert X_train.dtype == np.float32
    assert np.all(np.isfinite(X_train))
    logger.info("Creating the training model...")

    trainer = Trainer(
        model=ptychonn.model.ReconSmallPhaseModel(),
        batch_size=batch_size * torch.cuda.device_count(),
        output_path=iteration_out_path,
        output_suffix='',
    )

    trainer.setTrainingData(
        X_train,
        Y_train,
        valid_data_ratio=0.1,
    )

    trainer.setOptimizationParams(
        epochs_per_half_cycle=6,
        max_lr=5e-4,
        min_lr=1e-4,
    )
    trainer.initModel(model_params_path=load_model_path, )

    trainer.run(epochs)

    trainer.plotLearningRate(
        save_fname=iteration_out_path / 'learning_rate.svg',
        show_fig=False,
    )
    ptychonn.plot.plot_metrics(
        trainer.metrics,
        save_fname=iteration_out_path / 'metrics.svg',
        show_fig=False,
    )


class Trainer():
    '''
    '''

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
            shuffle=True,
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

        #Paper recommends 2-10 number of iterations
        self.epochs_per_half_cycle = epochs_per_half_cycle
        self.iters_per_half_cycle = epochs_per_half_cycle * self.iters_per_epoch

        logger.info(
            "LR step size is: %d which is every %d epochs",
            self.iters_per_half_cycle,
            self.iters_per_half_cycle / self.iters_per_epoch,
        )

        self.max_lr = max_lr
        self.min_lr = min_lr

        #criterion = lambda t1, t2: nn.L1Loss()
        self.criterion = self.customLoss
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

    # def testForwardSingleBatch(self):
    #     for ft_images, phs in self.trainloader:
    #         logger.info("batch size:", ft_images.shape)
    #         ph_train = self.model(ft_images)
    #         logger.info("Phase batch shape: ", ph_train.shape)
    #         logger.info("Phase batch dtype", ph_train.dtype)

    #         loss_ph = self.criterion(ph_train, phs, self.ntrain)
    #         logger.info("Phase loss", loss_ph)
    #         break

    def initModel(self, model_params_path: pathlib.Path | None = None):

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
            loss_p = self.criterion(pred_phs, phs, self.ntrain)
            # Monitor phase loss but only within support (which may not be same
            # as true amp)
            loss = loss_p
            # Use equiweighted amps and phase

            # Zero current grads and do backprop
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            tot_loss += loss.detach().item()

            loss_ph += loss_p.detach().item()

            # Update the LR according to the schedule -- CyclicLR updates each
            # batch
            self.scheduler.step()
            self.metrics['lrs'].append(self.scheduler.get_last_lr())
            self.scaler.update()

        self.metrics['losses'].append([tot_loss, loss_ph])

    def validate(self):
        tot_val_loss = 0.0
        val_loss_ph = 0.0
        for (ft_images, phs) in self.validloader:
            ft_images = ft_images.to(self.device)
            phs = phs.to(self.device)
            pred_phs = self.model(ft_images)  #Forward pass

            val_loss_p = self.criterion(pred_phs, phs, self.nvalid)
            val_loss = val_loss_p

            #try complex valued diff
            #diff_real = pred_amps * torch.cos(pred_phs) - amps * torch.cos(phs)
            #diff_imag = pred_amps * torch.sin(pred_phs) - amps * torch.sin(phs)
            #val_loss = torch.mean(torch.abs(diff_real + diff_imag))

            tot_val_loss += val_loss.detach().item()
            val_loss_ph += val_loss_p.detach().item()

        self.metrics['val_losses'].append([tot_val_loss, val_loss_ph])

        self.saveMetrics(self.metrics, self.output_path, self.output_suffix)

        # Update saved model if val loss is lower
        if (tot_val_loss < self.metrics['best_val_loss']):
            logger.info(
                "Saving improved model after Val Loss improved from %.5f to %.5f",
                self.metrics['best_val_loss'],
                tot_val_loss,
            )
            self.metrics['best_val_loss'] = tot_val_loss
            self.updateSavedModel(self.model, self.output_path,
                                  self.output_suffix)

    @staticmethod
    def customLoss(
        input,
        target,
        scaling,
    ):
        assert torch.all(torch.isfinite(input))
        assert torch.all(torch.isfinite(target))
        return torch.sum(torch.mean(
            torch.abs(input - target),
            axis=(-1, -2),
        )) / scaling

    @staticmethod
    #Function to update saved model if validation loss is minimum
    def updateSavedModel(
        model: ptychonn.model.ReconSmallPhaseModel,
        path: pathlib.Path,
        output_suffix: str = '',
    ):
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
        for epoch in range(epochs):

            #Set model to train mode
            self.model.train()

            #Training loop
            self.train()

            #Switch model to eval mode
            self.model.eval()

            #Validation loop
            self.validate()

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
                    'Epoch: %d | Ending LR: %.6f ',
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
