import xarray as xr
import numpy as np
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import PreparedGeometry, prep
import shapely.vectorized as vec

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

from plotter import Plotter
import pandas as pd


class Observer:
    """A class to handle observation utilities."""

    @classmethod
    def get_land(cls) -> PreparedGeometry:
        """From https://gist.github.com/pelson/9785576. It returns the land geometry on
        earth.

        Returns
        -------
        shapely.preprared.PreparedGeometry
            The land geometry.
        """

        land_shp_fname = shpreader.natural_earth(
            resolution="50m", category="physical", name="land"
        )
        land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
        return prep(land_geom)

    @staticmethod
    def _get_name(x: float, y: float) -> str:
        """It returns a standard name based on coordinates.

        Parameters
        ----------
        x: float
            The longitude.
        y: float
            The latutude.

        Returns
        -------
        str
            The standard name.
        """

        return f"lon{x}_lat{y}"

    @classmethod
    def generate_locations(
        cls,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        dx: float,
        dy: float,
        locs_output: str,
    ) -> tuple[np.ndarray, ...]:
        """It generates a grid of observation stations and writes them to the specified
        .xyn file for D-FlowFM.

        Parameters
        ----------
        x_min: float
            The minumum longitude.
        x_max: float
            The maximum longitude.
        y_min: float
            The minimum latitude.
        y_max: float
            The maximum latitude.
        dx: float
            The spacing for the grid in the x-direction.
        dy: float
            The spacing for the grid in the y-direction.
        locs_output: str
            The path of the output .xyn file.

        Returns
        -------
        numpy.ndarray
            The array of x-coordinates.
        numpy.ndarray
            The array of y-coordinates.
        numpy.ndarray
            The array of station names.
        """

        ys, xs = np.mgrid[y_min : y_max + dy : dy, x_min : x_max + dx : dx]
        land = cls.get_land()
        mask = vec.contains(land, xs, ys)

        flat_mask = mask.flatten()
        masked_xs = xs.flatten()[flat_mask].squeeze()
        masked_ys = ys.flatten()[flat_mask].squeeze()
        names = [cls._get_name(x, masked_ys[i]) for i, x in enumerate(masked_xs)]
        df = pd.DataFrame({"lon": masked_xs, "lat": masked_ys, "station": names})
        df.to_csv(locs_output, sep=" ", header=False, index=False)

        return xs, ys, mask.astype(float)


# # Test
# xs, ys, mask = Observer.generate_locations(-180, 180, -90, 90, 15, 15, "test.xyn")
# Plotter.plot_map(
#     xs, ys, mask, size=10, zorder_land=-1, cmap="bwr", path="land_grid.pdf"
# )
