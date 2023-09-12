import numpy as np
import xarray as xr
import pandas as pd


class DataHandler:
    """A class to homogenize the management and typing of data."""

    @classmethod
    def clip_dataset(cls, spin_up_steps: int, data: xr.Dataset) -> xr.Dataset:
        """It cuts a dataset to only inlcude data after the spin-up time.

        Parameters
        ----------
        spin_up_steps: int
            The estimated spin up steps.
        data: xarray.Dataset
            The datasets to clip.

        Returns
        -------
        xarray.Dataset
            The clipped datasets.
        """

        return data.isel(time=slice(spin_up_steps, None))

    @classmethod
    def read_noos(cls, path: str, s: slice = slice(None, None)) -> tuple[np.ndarray, np.ndarray]:
        """It reads a noos file.
        
        Parameters
        ----------
        path: str
            The path to the noos file.

        Returns
        -------
        numpy.ndarray
            The time array.
        numpy.ndarray
            The array of observations.
        """

        times = []
        values = []
        with open(path, "r") as file:
            for line in file:
                if not line.lstrip().startswith("#"):
                    time, value = line.split("\t")
                    times.append(time)
                    values.append(float(value))

            times = pd.to_datetime(times, format="%Y%m%d%H%M%S").to_numpy()
            values = np.array(values)
            return times[s], values[s]