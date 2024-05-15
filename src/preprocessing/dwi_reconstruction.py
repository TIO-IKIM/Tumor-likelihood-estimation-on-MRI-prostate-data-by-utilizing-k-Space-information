import numpy as np
from pathlib import Path
from tqdm import tqdm
from glob import glob
from utils import load_file_dwi, trapezoidal_regridding
from grappa import Grappa
import h5py
import logging

def undersample(kspace, factor: int):
    """ Skipping every nth kspace line

    Simulates acquiring every nth (where n is the acceleration factor) line
    of kspace, starting from the midline. Commonly used in SENSE algorithm.

    Parameters:
        kspace: Complex k-space numpy.ndarray with shape [coil, h, w]
        factor: Only scan every nth line (n=factor) starting from midline
    """
    if factor > 1:
        kspace = np.copy(kspace)
        mask = np.ones(kspace.shape, dtype=bool)
        midline = kspace.shape[1] // 2
        mask[:, midline::factor, :] = 0
        mask[:, midline::-factor, :] = 0
        kspace[mask] = 0

    return kspace

def grappa_recon(filepath: str) -> dict:
    """
    Perform GRAPPA reconstruction on DWI data.

    Args:
        filepath (str): The path to the DWI data file.

    Returns:
        dict: A dictionary containing the reconstructed k-space data for b50 and b1000.

    """
    kspace, calibration, coil_sens_maps, hdr = load_file_dwi(filepath)

    kspace_slice_regridded = trapezoidal_regridding(kspace[0, 0, ...], hdr)
    grappa_obj = Grappa(np.transpose(kspace_slice_regridded, (2, 0, 1)), kernel_size=(5, 5), coil_axis=1)

    grappa_weight_dict = {}
    for slice_num in range(kspace.shape[1]):
        calibration_regridded = trapezoidal_regridding(calibration[slice_num, ...], hdr)
        grappa_weight_dict[slice_num] = grappa_obj.compute_weights(
            np.transpose(calibration_regridded, (2, 0 ,1))
        )

    kspace_post_grappa = np.zeros(shape=kspace.shape, dtype=complex)

    for average in range(kspace.shape[0]):
        for slice_num in range(kspace.shape[1]):
            kspace_slice_regridded = trapezoidal_regridding(kspace[average, slice_num, ...], hdr)
            k = grappa_obj.apply_weights(
                np.transpose(kspace_slice_regridded, (2, 0, 1)),
                grappa_weight_dict[slice_num]
            )
            kspace_post_grappa[average][slice_num] = np.transpose(k, (1, 2, 0))

    kdict = {
        'b50x': np.sum(kspace_post_grappa[2:21:6, ...], axis=0) / 4,
        'b50y': np.sum(kspace_post_grappa[3:22:6, ...], axis=0) / 4,
        'b50z': np.sum(kspace_post_grappa[4:23:6, ...], axis=0) / 4,
        'b1000x': np.sum(
            np.r_[
                kspace_post_grappa[5:24:6, ...],
                kspace_post_grappa[26:48:3, ...]
            ], axis=0
        ) / 12,
        'b1000y': np.sum(
            np.r_[
                kspace_post_grappa[6:25:6, ...],
                kspace_post_grappa[27:49:3, ...]
            ], axis=0
        ) / 12,        
        'b1000z': np.sum(
            np.r_[
                kspace_post_grappa[7:26:6, ...],
                kspace_post_grappa[28:50:3, ...]
            ], axis=0
        ) / 12,
    }

    kspace_summed = {'b50': kdict['b50x'] + kdict['b50y'] + kdict['b50z'],
                 'b1000': kdict['b1000x'] + kdict['b1000y'] + kdict['b1000z']}
    
    return kspace_summed

def pca_compression(kspace):
    """
    Compresses multi-channel k-space using
    Principal Component Analysis.

        Parameters:
            kspace (ndarray): raw kspace with channels as first dimension
            
        Returns
            compressed kspace (nd-array)
    """
    
    # reshape as 2D matrix
    orig_size = list(kspace.shape)
    tmp = np.reshape(kspace, [orig_size[0], np.prod(orig_size[1:])])
    
    # Do singular value decomposition (SVD)
    u, sigma, vh = np.linalg.svd(np.conj(tmp).T, full_matrices=False)
    
    # do basis transform
    tmp = np.matmul(vh, tmp)
            
    # compute variances and determine number of relevant channels
    c = sigma**2
    num_coils = len(np.where(c>0.05*c[0])[0])
    
    # extract relevant channels
    tmp = tmp[:num_coils, :]
    
    # reshape
    orig_size[0] = num_coils
    finalKspace = np.reshape(tmp, orig_size)
    
    # pass transformed data
    return finalKspace

def run(input_path: str, save_path: str, grappa: bool=False, averaging: bool=True):
    """
    Perform PCA-based reconstruction on DWI data.

    Args:
        input_path (str): Path to the input data.
        save_path (str): Path to save the reconstructed data.
        grappa (bool): Flag indicating whether to use GRAPPA reconstruction.
        averaging (bool): Flag indicating whether to average slices before PCA.

    Returns:
        None
    """

    train_list = glob(input_path + "/*")

    for file in tqdm(train_list):
        if grappa:
            kspace_grappa_dict = grappa_recon(file)
            filename = Path(file).stem

            kspace_summed = kspace_grappa_dict['b50'] + kspace_grappa_dict['b1000']

            for slc in range(kspace_summed.shape[0]):
                x_pca = pca_compression(kspace_summed[slc, ...])[0,...]
                np.save(f"{Path(save_path, filename)}_slice{slc + 1}.npy", x_pca)

        else:
            kspace = h5py.File(file, 'r')['kspace']
            filename = Path(file).stem
            if averaging:
                logging.info("Averaging slices")
                kspace_summed = np.sum(kspace, axis=0)
                logging.info("Slices averaged. Perform slicewise PCA.")
                for slc in range(kspace_summed.shape[0]):
                    kspace_summed_slc = kspace_summed[slc, ...]
                    x_pca = pca_compression(kspace_summed_slc)[0,...]
                    np.save(f"{Path(save_path, filename)}_slice{slc + 1}.npy", x_pca)
            else:
                for slc in range(kspace.shape[1]):
                    kspace_slice = kspace[:,slc,...].swapaxes(0, 1)
                    x_pca = np.empty(kspace_slice.shape[1:], dtype=complex)
                    for i in range(kspace_slice.shape[1]):
                        x_pca[i, ...] = pca_compression(kspace_slice[:,i,...])[0]
                    np.save(f"{Path(save_path, filename)}_slice{slc + 1}.npy", x_pca)

if __name__ == "__main__":
    train_path = "/path/to/input"
    save_path = "/path/to/output"

    grappa = False
    averaging = True

    run(train_path, save_path, grappa, averaging)