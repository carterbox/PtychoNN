import pathlib
import importlib.resources

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import torch
import tqdm

import ptychonn._model as helper_small_model


def stitch_from_inference(
    inferences: np.array,
    scan: np.array,
    pix: int = 0,
    stitched_pixel_width: float = 10e-9,
    inference_pixel_width: float = 10e-9,
) -> np.array:
    '''Combine many overlapping inferred patches into a single image.

    Parameters
    ----------
    inferences : (POSITION, WIDTH, HEIGHT)
        Overlapping patches of the inferred field of view.
    scan : (POSITION, 2)
        The relative coordinates of each of the overlapping patches.
    pix : int
        Shrink the interpolation region by this number of pixels.
    stitched_pixel_width : float [m]
        The width of a pixel in the stitched image.
    inference_pixel_width : float [m]
        The width of a pixel in the inferred image patches.

    Returns
    -------
    stitched : (COMBINED_WIDTH, COMBINED_HEIGHT) np.array
        The stitched together image.
    '''
    pos_x = scan[..., 0]
    pos_y = scan[..., 1]

    # The global axes of the stitched image in meters
    x = np.arange(pos_x.min(),
                  pos_x.max() + inferences.shape[-1] * inference_pixel_width,
                  step=stitched_pixel_width)
    y = np.arange(pos_y.min(),
                  pos_y.max() + inferences.shape[-1] * inference_pixel_width,
                  step=stitched_pixel_width)

    result = np.zeros((y.shape[0], x.shape[0]))
    cnt = np.zeros_like(result)

    # The local axes of the inference patches meters
    xx = np.arange(inferences.shape[-1]) * inference_pixel_width

    for i in tqdm.tqdm(range(len(inferences)), leave=False):
        data_ = inferences[i]
        xxx = xx + pos_x[i]
        yyy = xx + pos_y[i]

        if pix > 0:
            xxx = xxx[pix:-pix]
            yyy = yyy[pix:-pix]
            data_ = data_[pix:-pix, pix:-pix]
        find_pha = scipy.interpolate.interp2d(
            xxx,
            yyy,
            data_,
            kind='linear',
            fill_value=0,
        )
        tmp = find_pha(x, y)
        cnt += tmp != 0
        result += tmp

    return result / np.maximum(cnt, 1)


@click.command(name='infer')
@click.argument(
    'data_path',
    type=click.Path(
        exists=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    'scan_path',
    type=click.Path(
        exists=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    'out_dir',
    type=click.Path(
        exists=True,
        path_type=pathlib.Path,
    ),
)
def infer_cli(
    data_path: pathlib.Path,
    scan_path: pathlib.Path,
    out_dir: pathlib.Path,
):
    '''Infer a reconstructed image from diffraction patterns at DATA_PATH and
    scan positions at SCAN_PATH. Save inferred patches and stitched image at
    OUT_DIR.
    '''

    inferences_out_file = out_dir / 'inferences_506.npz'
    click.echo(f'Does data path exist? {data_path.exists()}')

    with h5py.File(data_path) as f:
        data = f['entry/data/data'][()]

    inferences = infer(
        data=data,
        inferences_out_file=inferences_out_file,
    )

    ## parameters required for stitching individual inferences
    spiral_step = 0.05
    step = spiral_step * -1e-6
    spiral_traj = np.load(scan_path)
    scan = np.stack((spiral_traj['x'], spiral_traj['y']), axis=-1) * step
    stitched = stitch_from_inference(inferences, scan)

    # Plotting some summary images
    plt.figure(1, figsize=[8.5, 7])
    plt.pcolormesh(stitched)
    plt.colorbar()
    plt.tight_layout()
    plt.title('stitched_inferences')
    plt.savefig(out_dir / 'stitched_506.png', bbox_inches='tight')

    test_inferences = [0, 1, 2, 3]
    fig, axs = plt.subplots(1, 4, figsize=[13, 3])
    for ix, inf in enumerate(test_inferences):
        plt.subplot(1, 4, ix + 1)
        plt.pcolormesh(inferences[inf])
        plt.colorbar()
        plt.title('Inference at position {0}'.format(inf))
    plt.tight_layout()
    plt.savefig(out_dir / 'inferences_0_to_4_scan506.png', bbox_inches='tight')

    return 0


def infer(
    data: np.array,
    *,
    inferences_out_file: pathlib.Path = None,
) -> np.array:
    '''Infer ptychography reconstruction for the given data.

    For each diffraction pattern in data, the corresponding patch of object is
    reconstructed.

    Set the CUDA_VISIBLE_DEVICES environment variable to control which GPUs
    will be used.

    Parameters
    ----------
    data : (POSITION, WIDTH, HEIGHT)
        Diffraction patterns to be reconstructed.
    inferences_out_file : pathlib.Path
        Optional file to save reconstructed patches.

    Returns
    -------
    inferences : (POSITION, WIDTH, HEIGHT)
        The reconstructed patches inferred by the model.
    '''

    with importlib.resources.path(
            'ptychonn._infer',
            'weights.pth',
    ) as load_model_path:
        click.echo(f'Loading model at {load_model_path}')
        recon_model = helper_small_model.ReconSmallPhaseModel()
        tester = helper_small_model.Tester(
            model=recon_model,
            batch_size=max(torch.cuda.device_count(), 1) * 64,
            model_params_path=load_model_path,
        )

    tester.setTestData(data)
    inferences = tester.predictTestData(npz_save_path=inferences_out_file)

    return inferences
