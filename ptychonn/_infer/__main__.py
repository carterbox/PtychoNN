import importlib.resources
import pathlib
import typing
import glob

from torch.utils.data import TensorDataset, DataLoader
import click
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.interpolate
import torch
import tqdm

import ptychonn.model


def stitch_from_inference(
    inferences: npt.NDArray,
    scan: npt.NDArray,
    *,
    stitched_pixel_width: float,
    inference_pixel_width: float,
    pix: int = 0,
) -> npt.NDArray:
    '''Combine many overlapping inferred patches into a single image.

    Parameters
    ----------
    inferences : (POSITION, WIDTH, HEIGHT)
        Overlapping patches of the inferred field of view.
    scan : (POSITION, 2) [m]
        The coordinates of each of the overlapping patches.
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
    pos_x = scan[..., 1]
    pos_y = scan[..., 0]

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
    'data_dir',
    type=click.Path(
        exists=True,
        path_type=pathlib.Path,
    ),
)
@click.argument(
    'params_path',
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
    data_dir: pathlib.Path,
    params_path: pathlib.Path,
    out_dir: pathlib.Path,
):
    '''Infer a reconstructed image from diffraction patterns at DATA_PATH and
    scan positions at SCAN_PATH. Save inferred patches and stitched image at
    OUT_DIR.
    '''

    dataslist = []
    scanlist = []

    for name in glob.glob(str(data_dir / '*.npz')):
        print(name)
        with np.load(name) as f:
            dataslist.append(f['reciprocal'])
            scanlist.append(f['scan'])

    data = np.concatenate(dataslist, axis=0)
    scan = np.concatenate(scanlist, axis=0)

    inferences = infer(
        data=data,
        model_params_path=params_path,
    )

    pstitched = stitch_from_inference(
        inferences[:, 0],
        scan,
        stitched_pixel_width=1,
        inference_pixel_width=1,
    )
    astitched = stitch_from_inference(
        inferences[:, 1],
        scan,
        stitched_pixel_width=1,
        inference_pixel_width=1,
    )

    # Plotting some summary images
    plt.figure(1, figsize=[8.5, 7])
    plt.imshow(pstitched)
    plt.colorbar()
    plt.tight_layout()
    plt.title('stitched_phases')
    plt.savefig(out_dir / 'pstitched.png', bbox_inches='tight')

    plt.figure(2, figsize=[8.5, 7])
    plt.imshow(astitched)
    plt.colorbar()
    plt.tight_layout()
    plt.title('stitched_amplitudes')
    plt.savefig(out_dir / 'astitched.png', bbox_inches='tight')

    test_inferences = [0, 1, 2, 3]
    fig, axs = plt.subplots(1, 4, figsize=[13, 3])
    for ix, inf in enumerate(test_inferences):
        plt.subplot(2, 4, ix + 1)
        plt.pcolormesh(inferences[inf, 0])
        plt.colorbar()
        plt.title('Inference at position {0}'.format(inf))
        plt.subplot(2, 4, 4 + ix + 1)
        plt.pcolormesh(inferences[inf, 1])
        plt.colorbar()
        plt.title('Inference at position {0}'.format(inf))
    plt.tight_layout()
    plt.savefig(out_dir / 'inferences.png', bbox_inches='tight')

    return 0


def infer(
    data: npt.NDArray,
    model_params_path: pathlib.Path,
    *,
    inferences_out_file: typing.Optional[pathlib.Path] = None,
) -> npt.NDArray:
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
    inferences : (POSITION, 2, WIDTH, HEIGHT)
        The reconstructed patches inferred by the model.
    '''
    tester = Tester(
        model=ptychonn.model.ReconSmallModel(),
        model_params_path=model_params_path,
    )
    tester.setTestData(
        data,
        batch_size=max(torch.cuda.device_count(), 1) * 64,
    )
    return tester.predictTestData(npz_save_path=inferences_out_file)


class Tester():

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        model_params_path: pathlib.Path,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Let's use {torch.cuda.device_count()} GPUs!")

        self.model = model

        params = torch.load(
            model_params_path,
            map_location=self.device,
        )
        self.model.load_state_dict(params)

        self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)

        self.model.eval()

    def setTestData(self, X_test: np.ndarray, batch_size: int):
        self.X_test = torch.tensor(X_test[:, None, ...], dtype=torch.float32)
        self.test_data = TensorDataset(self.X_test)
        self.testloader = DataLoader(
            self.test_data,
            batch_size=batch_size,
            shuffle=False,
        )

    def predictTestData(self, npz_save_path: str = None):

        phs_eval = []
        with torch.inference_mode():
            for (ft_images, ) in self.testloader:
                ph_eval = self.model(ft_images.to(self.device))
                phs_eval.append(ph_eval.detach().cpu().numpy())

        self.phs_eval = np.concatenate(phs_eval, axis=0)

        if npz_save_path is not None:
            np.savez_compressed(npz_save_path, ph=self.phs_eval)
            print(f'Finished the inference stage and saved at {npz_save_path}')

        return self.phs_eval

    def calcErrors(self, phs_true: np.ndarray, npz_save_path: str = None):
        from skimage.metrics import mean_squared_error as mse

        self.phs_true = phs_true
        self.errors = []
        for i, (p1, p2) in enumerate(zip(self.phs_eval, self.phs_true)):
            err2 = mse(p1, p2)
            self.errors.append([err2])

        self.errors = np.array(self.errors)
        print("Mean errors in phase")
        print(np.mean(self.errors, axis=0))

        if npz_save_path is not None:
            np.savez_compressed(npz_save_path, phs_err=self.errors[:, 0])

        return self.errors
