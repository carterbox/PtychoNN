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
    out_dir='simulated',
):

    os.makedirs(out_dir, exist_ok=True)

    shape = S
    lambda0 = 1.24e-9 / 10
    dx_dec = 75e-6
    dis_defocus = 800e-6
    dis_StoD = 2
    dx = lambda0 * dis_StoD / shape / dx_dec

    # test single probe modes
    probe = tike.ptycho.fresnel.single_probe(shape,
                                             lambda0,
                                             dx,
                                             dis_defocus,
                                             zone_plate_params=dict(
                                                 radius=150e-6 / 2,
                                                 outmost=50e-9,
                                                 beamstop=60e-6,
                                             ))

    plt.imsave(f'{out_dir}/probe-amplitude.png', np.abs(probe[0, 0, 0]))
    plt.imsave(f'{out_dir}/probe-phase.png',
               np.angle(probe[0, 0, 0]),
               vmin=-np.pi,
               vmax=np.pi,
               cmap=plt.cm.twilight)

    for a, b in itertools.permutations(libimage.available_images[:3], 2):

        phase = (libimage.load(a, width) - 0.5).astype(np.csingle)
        amplitude = (1.0 - libimage.load(b, width)).astype(np.csingle)
        fov = (amplitude * np.exp(1j * phase * np.pi))
        plt.imsave(f'{out_dir}/phase-{a}.png',
                   np.angle(fov),
                   vmin=-np.pi,
                   vmax=np.pi,
                   cmap=plt.cm.twilight)
        plt.imsave(f'{out_dir}/amplitude-{b}.png', np.abs(fov))
        assert np.abs(fov).max(
        ) <= 1.001, f'f{np.abs(fov).max()} is larger than 1.0 for {b}'
        assert np.abs(fov).min(
        ) >= 0.0, f'f{np.abs(fov).min()} is smaller than 0.0 for {b}'
        assert np.angle(fov).max() <= +np.pi, a
        assert np.angle(fov).min() >= -np.pi, a

        scan = np.random.uniform(1, width - S - 1, (num_scan, 2))

        patches = tike.ptycho.learn.extract_patches(
            psi=fov,
            scan=scan,
            patch_width=probe.shape[-1],
        ).astype('complex64')
        print(patches.dtype, patches.shape)

        diffraction = np.fft.ifftshift(
            tike.ptycho.simulate(
                detector_shape=probe.shape[-1],
                probe=probe,
                scan=scan,
                psi=fov,
            ),
            axes=(-2, -1),
        ).astype('float32')
        print(diffraction.dtype, diffraction.shape)
        plt.imsave(f'{out_dir}/diffraction-{a}-{b}.png',
                   diffraction[num_scan // 2])

        print(f'Training params = {np.prod(diffraction.shape)}')

        np.savez_compressed(
            f'{out_dir}/simulated-{a}-{b}.npz',
            reciprocal=diffraction,
            real=patches,
        )


if __name__ == '__main__':
    test_construct_simulated_training_set()
