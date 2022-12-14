import itertools
import os

import libimage
import numpy as np
import tifffile
import tike.ptycho
import tike.ptycho.learn


def test_construct_simulated_training_set(
    W=2048,
    N=1024,
    S=128,
    out_dir='simulated',
):

    os.makedirs(out_dir, exist_ok=True)

    probe = (tike.ptycho.probe.gaussian(S, 0.6, 0.9)[None, None, None, ...] *
             np.exp(1j * np.pi *
                    (np.random.rand(S, S) - 0.5))).astype('complex64')

    tifffile.imwrite(f'{out_dir}/probe-amplitude.tiff', np.abs(probe[0, 0, 0]))
    tifffile.imwrite(f'{out_dir}/probe-phase.tiff', np.angle(probe[0, 0, 0]))

    for a, b in itertools.permutations(libimage.available_images, 2):

        phase = libimage.load(a, W) - 0.5
        amplitude = 1 - libimage.load(b, W)
        fov = (amplitude * np.exp(1j * phase * np.pi)).astype('complex64')
        tifffile.imwrite(f'{out_dir}/phase-{a}.tiff', np.angle(fov))
        tifffile.imwrite(f'{out_dir}/amplitude-{b}.tiff', np.abs(fov))
        assert np.abs(fov).max() <= 1.0
        assert np.abs(fov).min() >= 0.0

        scan = np.random.uniform(1, W - S - 1, (N, 2))

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
        tifffile.imwrite(f'{out_dir}/diffraction-{a}-{b}.tiff',
                         diffraction[N // 2])

        print(f'Training params = {np.prod(diffraction.shape)}')

        np.savez_compressed(
            f'{out_dir}/simulated-{a}-{b}.npz',
            reciprocal=diffraction,
            real=patches,
        )


if __name__ == '__main__':
    test_construct_simulated_training_set()
