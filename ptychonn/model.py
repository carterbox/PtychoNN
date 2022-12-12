import numpy as np
import torch
import torch.nn as nn


class Multiply(nn.Module):
    """Just multiplies the tensor by a constant."""

    def __init__(self, constant: float) -> None:
        super().__init__()
        self.constant = constant

    def forward(self, x):
        return x * self.constant


class ReconSmallPhaseModel(nn.Module):
    """

    Parameters
    ----------
    nconv: int
        The number of filters in the first and last convolutional layers
    use_batch_norm: bool
        Whether to use batch normalization before activation layers
    min_shape: int
        The minimum required shape for the input diffraction patterns

    """

    def __init__(
        self,
        nconv: int = 16,
        use_batch_norm=False,
        with_amplitude=False,
        with_phase=True,
    ):
        super(ReconSmallPhaseModel, self).__init__()

        # The minimum shape is a result of the number of pooling operations
        self._min_shape = 16
        self._with_phase = with_phase
        self._with_amplitude = with_amplitude

        self.encoder = nn.Sequential(
            *self.down_block(1, nconv, use_batch_norm),
            *self.down_block(nconv, nconv * 2, use_batch_norm),
            *self.down_block(nconv * 2, nconv * 4, use_batch_norm),
            *self.down_block(nconv * 4, nconv * 8, use_batch_norm),
        )

        if with_amplitude:
            self.decoder_amplitude = nn.Sequential(
                *self.up_block(nconv * 8, nconv * 8, use_batch_norm),
                *self.up_block(nconv * 8, nconv * 4, use_batch_norm),
                *self.up_block(nconv * 4, nconv * 2, use_batch_norm),
                *self.up_block(nconv * 2, nconv * 1, use_batch_norm),
                nn.Conv2d(nconv * 1, 1, 3, stride=1, padding=(1, 1)),
            )

        if with_phase:
            self.decoder_phase = nn.Sequential(
                *self.up_block(nconv * 8, nconv * 8, use_batch_norm),
                *self.up_block(nconv * 8, nconv * 4, use_batch_norm),
                *self.up_block(nconv * 4, nconv * 2, use_batch_norm),
                *self.up_block(nconv * 2, nconv * 1, use_batch_norm),
                nn.Conv2d(nconv * 1, 1, 3, stride=1, padding=(1, 1)),
                nn.BatchNorm2d(1) if use_batch_norm else torch.nn.Identity(),
                nn.Tanh(),
                # Restore -pi to pi range using tanh activation (-1 to 1) for
                # phase and multiplying by pi
                Multiply(np.pi),
            )

    @property
    def min_shape(self):
        return self._min_shape

    @property
    def with_amplitude(self):
        return self._with_amplitude

    @property
    def with_phase(self):
        return self._with_phase

    def down_block(self, filters_in, filters_out, use_batch_norm):
        return [
            nn.Conv2d(
                in_channels=filters_in,
                out_channels=filters_out,
                kernel_size=3,
                stride=1,
                padding=(1, 1),
            ),
            nn.BatchNorm2d(filters_out)
            if use_batch_norm else torch.nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(filters_out, filters_out, 3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(filters_out)
            if use_batch_norm else torch.nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        ]

    def up_block(self, filters_in, filters_out, use_batch_norm):
        return [
            nn.Conv2d(
                filters_in,
                filters_out,
                3,
                stride=1,
                padding=(1, 1),
            ),
            nn.BatchNorm2d(filters_out)
            if use_batch_norm else torch.nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(filters_out, filters_out, 3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(filters_out)
            if use_batch_norm else torch.nn.Identity(),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')
        ]

    def forward(self, x):
        assert x.shape[-1] >= self.min_shape
        assert x.shape[-2] >= self.min_shape
        with torch.cuda.amp.autocast():
            return self.decoder_phase(self.encoder(x))
