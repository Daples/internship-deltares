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
        """From https://gist.github.com/pelson/9785576."""

        land_shp_fname = shpreader.natural_earth(
            resolution="50m", category="physical", name="land"
        )
        land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
        return prep(land_geom)

    @staticmethod
    def _get_name(x: float, y: float) -> str:
        """It returns a standard name based on coordinates."""

        return f"lon{x}_lat{y}"

    # @classmethod
    # def is_land(cls, x: float, y: float) -> bool:
    #     return land.contains(sgeom.Point(x, y))

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


xs, ys, mask = Observer.generate_locations(-180, 180, -90, 90, 15, 15, "test.xyn")
# Plotter.plot_map(
#     xs, ys, mask, size=10, zorder_land=-1, cmap="bwr", path="land_grid.pdf"
# )

# plt.figure(figsize=(12, 12))
# ax = plt.axes(projection=ccrs.PlateCarree())
# cmap = mcolors.ListedColormap(["blue", "green"])


# ax.scatter(
#     xs,
#     ys,
#     s=(mask + 0.5) * 20,
#     c=mask + 1,
#     cmap=cmap,
#     edgecolor="none",
#     transform=ccrs.PlateCarree(),
# )

# ax.add_geometries(
#     [Observer.get_land()], ccrs.PlateCarree(), facecolor="none", edgecolor="black"
# )
# plt.show()
