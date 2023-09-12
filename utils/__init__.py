import numpy as np


def get_rmses(
    data_full: np.ndarray,
    ts_clipped: np.ndarray,
    data_clipped: np.ndarray,
) -> np.ndarray:
    """It returns the RMSE for all possible windows.

    Parameters
    ----------
    data_full: numpy.ndarray
        The array of full data.
    ts_clipped: numpy.ndarray
        The array of clipped times (times of the late/second simulation).
    data_clipped: numpy.ndarray
        The array of clipped data.

    Returns
    -------
    numpy.ndarray
        The RMSE per number of data points excluded.
    """

    len_clip = ts_clipped.shape[0]
    clipped_data_full = data_full[-len_clip:]

    rmses = np.zeros(len_clip - 1)
    for h in range(len_clip - 1):
        rmses[h] = np.std(clipped_data_full[h:] - data_clipped[h:], ddof=1)

    return rmses


def get_diagnostics(array: np.ndarray) -> np.ndarray:
    """It prints general diagnostics of a data array and returns the indices of columns
    that are not all NaN.

    Parameters
    ----------
    array: numpy.ndarray
        The data array.

    Returns
    -------
    numpy.ndarray
        The bool array of whether a column is not all NaN.
    """

    idxs = np.arange(array.shape[1])
    r = np.isnan(array).all(axis=0)
    all_nans = idxs[r]
    any_nans = idxs[np.isnan(array).any(axis=0)]
    print(f"shape: {array.shape}")
    print(f"n_cols with any nans: {any_nans.size}, idxs: {any_nans}")
    print(f"n_cols with all nans: {all_nans.size}, idxs: {all_nans}")
    return ~r


def Create_Pmsl(gConst=10.813, rhow=1023, bias=0.0):
    # This bias changes the bias in m to pressure (Pa)
    # gConst is the gravitational constant
    # rhow is the density of water
    # bias is in meters [m]
    return -1 * bias * gConst * rhow
