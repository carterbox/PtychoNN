import itertools
import os

import libimage
import numpy as np
import matplotlib.pyplot as plt
import tike.ptycho
import tike.ptycho.learn


def test_construct_simulated_training_set(
    width=1024,
    num_scan=1024,
    S=64,
    out_dir='imagenet',
):

    os.makedirs(out_dir, exist_ok=True)

    shape = S
    lambda0 = 1.24e-9 / 10
    dx_dec = 75e-6
    dis_defocus = 800e-6
    dis_StoD = 2
    dx = lambda0 * dis_StoD / shape / dx_dec

    # test single probe modes
    probe = tike.ptycho.fresnel.single_probe(
        shape,
        lambda0,
        dx,
        dis_defocus,
        zone_plate_params=dict(
            radius=150e-6 / 2,
            outmost=50e-9,
            beamstop=60e-6,
        ),
    )

    plt.imsave(
        f'{out_dir}/probe-amplitude.png',
        np.abs(probe[0, 0, 0]),
    )
    plt.imsave(
        f'{out_dir}/probe-phase.png',
        np.angle(probe[0, 0, 0]),
        vmin=-np.pi,
        vmax=np.pi,
        cmap=plt.cm.twilight,
    )

    probe = probe[0, 0, 0]

    with np.load(
            f"{os.environ['HOME']}/Documents/fluffy-umbrella/test/Imagenet64/train_data_batch_1.npz"
    ) as f:
        patches = f['data'].reshape(-1, 3, 64, 64)[:, 0] / 255.0 - 0.5
        print(patches.shape, patches.dtype, patches.min(), patches.max())

    angle = np.pi * patches

    psi = (1.0 * np.exp(1j * angle)).astype(np.complex64)

    wavefront = (probe * psi).astype(np.complex64)

    diffraction = np.fft.ifftshift(
        np.square(np.abs(np.fft.fft2(wavefront))),
        axes=(-2, -1),
    ).astype(np.float32)
    print(diffraction.dtype, diffraction.shape)

    samples = np.linspace(0, num_scan, num=5, dtype=int, endpoint=True)

    plt.imsave(
        f'{out_dir}/diffraction-imagenet.png',
        np.concatenate(diffraction[samples], axis=-1),
        cmap=plt.cm.gray,
    )
    plt.imsave(
        f'{out_dir}/angle-imagenet.png',
        np.concatenate(angle[samples], axis=-1),
        vmin=-np.pi,
        vmax=+np.pi,
        cmap=plt.cm.twilight,
    )

    print(f'Training params = {np.prod(diffraction.shape)}')

    np.savez(
        f'{out_dir}/simulated-imagenet.npz',
        reciprocal=diffraction,
        real=psi,
    )


if __name__ == '__main__':
    test_construct_simulated_training_set()
