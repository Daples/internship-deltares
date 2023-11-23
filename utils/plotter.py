import os
from typing import Any, Union, Callable, cast

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.dates as mdates
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.crs import Projection


class Plotter:
    """A class to wrap the plotting functions.

    (Static) Attributes
    -------------------
    _folder: str
        The folder to store the output figures.
    args: list[Any]
        The additional arguments for all plots.
    kwargs: dict[str, Any]
        The keyword arguments for all plots.
    """

    _figs_folder: str = "figs"
    _folder: Callable[[str], str] = lambda folder: os.path.join(os.getcwd(), folder)
    args: list[Any] = ["k-"]
    kwargs: dict[str, Any] = {"markersize": 3}
    figsize_standard: tuple[int, int] = (8, 5)
    figsize_map: tuple[int, int] = (14, 5)
    figsize_compare_map: tuple[int, int] = (12, 5)
    figsize_horizontal: tuple[int, int] = (16, 5)
    figsize_vertical: tuple[int, int] = (8, 10)
    font_size: int = 14
    bands_alpha: float = 0.2
    grid_alpha: float = 0.7
    ensemble_alpha: float = 0.2
    grid_color: str = "lightgray"
    dpi: int = 200
    h_label: str = "$h\ (\mathrm{m})$"
    u_label: str = "$u\ (\mathrm{m})$"
    x_label: str = "$x\ (\mathrm{km})$"
    c_label: str = "$c\ (\mathrm{m/s}))$"
    t_label: str = "Time"
    display_projection: Projection = ccrs.Robinson()
    orig_projection: Projection = ccrs.PlateCarree()
    color_map: str = "viridis"
    land_color: str = "antiquewhite"
    coast_resolution: str = "50m"
    coast_width: float = 0.5
    station_color: str = "deeppink"

    @classmethod
    def change_folder(cls, new_folder: str) -> None:
        """It changes the default figs folder name.

        Parameters
        ----------
        new_folder: str
            The new folder name.
        """

        cls._figs_folder = new_folder

    @staticmethod
    def __clear__() -> None:
        """It clears the graphic objects."""

        plt.close("all")
        plt.cla()
        plt.clf()

    @classmethod
    def __setup_config__(cls) -> None:
        """It sets up the matplotlib configuration."""

        plt.rc("text", usetex=True)
        plt.rcParams.update({"font.size": cls.font_size})

    @classmethod
    def legend(cls, ax: Axes) -> None:
        """It moved the legend outside the plot.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            The axes.
        """

        ax.legend(
            bbox_to_anchor=(0, 1, 1, 0.2),
            loc="upper right",
            ncol=4,
        )

    @classmethod
    def date_axis(cls, ax: Axes) -> None:
        """It formats the x-axis for dates.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            The axes.
        """

        plt.setp(ax.get_xticklabels(), rotation=30, fontsize=cls.font_size)
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
        )

    @classmethod
    def grid(cls, ax: Axes) -> None:
        """It adds a grid to the axes.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            The axes.
        """

        ax.grid(alpha=cls.grid_alpha, color=cls.grid_color)

    @classmethod
    def map_grid(cls, ax: GeoAxes) -> None:
        """It adds the geographic grid to the axes.

        Parameters
        ----------
        cartopy.mpl.geoaxes.GeoAxes
            The geographic axes.
        """

        gl = ax.gridlines(
            draw_labels=True,
            dms=True,
            x_inline=False,
            y_inline=False,
            alpha=cls.grid_alpha,
            color=cls.grid_color,
        )
        gl.top_labels = False
        gl.right_labels = False

    @classmethod
    def add_folder(cls, path: str) -> str:
        """It adds the default folder to the input path.

        Parameters
        ----------
        path: str
            A path in string.

        Returns
        -------
        str
            The path with the added folder.
        """

        check = cls._folder(cls._figs_folder)
        if not os.path.exists(check):
            os.mkdir(check)

        return os.path.join(check, path)

    @staticmethod
    def get_sizes(z: np.ndarray, size_lims: tuple[float, float]) -> np.ndarray:
        """It returns the scatter marker sizes. The lowest is the largest.

        Parameters
        ----------
        z: numpy.ndarray
            The axis data.
        size_lims: tuple[float, float]
            The lower and upper sizes for the markers.

        Returns
        -------
        numpy.ndarray
            The array of marker sizes.
        """

        z_size = z - z.min()
        z_size /= z_size.max()
        inv_z = z_size.max() - z_size
        size = cast(np.ndarray, inv_z) * size_lims[1] + size_lims[0]
        return size

    @classmethod
    def mplot(
        cls,
        x,
        ys,
        labels: Union[list[str], None] = None,
        path: Union[str, None] = None,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
        clear: bool = True,
        is_ax_date: bool = False,
        linewidth: float = 1,
    ) -> tuple[Figure, Axes]:
        """It plots several lines with standard formatting.

        Parameters
        ----------
        x
            The data on horizontal axis.
        ys
            The data sets on vertical axis. Shape: (data, samples)
        labels: list[str] | None
            The labels for each data sample (legend). Default: None
        path: str | None, optional
            The name to save the figure with. Default: None
        xlabel: str, optional
            The label of the horizontal axis. Default: "$x$"
        ylabel: str, optional
            The label of the vertical axis. Default: "$y$"
        clear: bool
            Whether to clear the figure or not. Default: True
        is_ax_date: bool
            Whether the horizontal axis should be date formatted. Default: False
        linewidth: float
            The line width for the plots. Default: 1

        Returns
        -------
        matplotlib.figure.Figure
            The figure handle.
        matplotlib.figure.Axes
            The axes handle.
        """

        if clear:
            cls.__clear__()
            cls.__setup_config__()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=cls.figsize_standard)
        kwargs = {"linewidth": linewidth}
        if labels is not None:
            kwargs = {"label": labels}

        for i in range(ys.shape[1]):
            ax.plot(x, ys[:, i], **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if is_ax_date:
            cls.date_axis(ax)
        cls.grid(ax)
        if path is not None:
            plt.savefig(cls.add_folder(path), bbox_inches="tight")
        return fig, ax

    @classmethod
    def plot_bands(
        cls,
        x,
        y,
        s,
        path: Union[str, None] = None,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
        clear: bool = True,
        ensure_positive: bool = False,
        n_bands: int = 1,
    ) -> tuple[Figure, Axes]:
        """It creates a bands plot with standard formatting.

        Parameters
        ----------
        x
            The data on horizontal axis.
        y
            The data on vertical axis.
        s
            The dispersion of each data point. Bands are filled in [y - s, y + s].
        path: str | None, optional
            The name to save the figure with. Default: None
        xlabel: str, optional
            The label of the horizontal axis. Default: "$x$"
        ylabel: str, optional
            The label of the vertical axis. Default: "$y$"
        clear: bool
            Whether to clear the figure or not. Default: True
        ensure_positive: bool, optional
            Whether the value is a non-negative number. Defalut: False
        n_bands: int, optional
            The number of bands to plot. Default: 1

        Returns
        -------
        matplotlib.figure.Figure
            The figure handle.
        matplotlib.figure.Axes
            The axes handle.
        """

        func = lambda x: x
        if ensure_positive:
            func = lambda x: np.maximum(x, 0)

        if clear:
            cls.__clear__()
            cls.__setup_config__()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=cls.figsize_standard)
        ax.plot(x, y, *cls.args)

        for i in range(1, n_bands + 1):
            ax.fill_between(
                x,
                func(y - i * s),
                func(y + i * s),
                color="b",
                alpha=cls.bands_alpha / (i + 1),
                zorder=-1,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cls.grid(ax)
        if path is not None:
            plt.savefig(cls.add_folder(path), bbox_inches="tight")
        return fig, ax

    @classmethod
    def plot(
        cls,
        x,
        y,
        path: Union[str, None] = None,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
        clear: bool = True,
    ) -> tuple[Figure, Axes]:
        """It creates a plot with standard formatting.

        Parameters
        ----------
        x
            The data on horizontal axis.
        y
            The data on vertical axis.
        path: str | None, optional
            The name to save the figure with. Default: None
        xlabel: str, optional
            The label of the horizontal axis. Default: "$x$"
        ylabel: str, optional
            The label of the vertical axis. Default: "$y$"
        clear: bool
            Whether to clear the figure or not. Default: True

        Returns
        -------
        matplotlib.figure.Figure
            The figure handle.
        matplotlib.figure.Axes
            The axes handle.
        """

        if clear:
            cls.__clear__()
            cls.__setup_config__()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=cls.figsize_standard)
        ax.plot(x, y, *cls.args, **cls.kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cls.grid(ax)
        if path is not None:
            plt.savefig(cls.add_folder(path), bbox_inches="tight")
        return fig, ax

    @classmethod
    def plot_series(
        cls,
        ts_list: list,
        data_list: list[np.ndarray],
        loc_name: str,
        title_func: Callable,
        legends: Union[list[str], None] = None,
        obs_data: Union[list[np.ndarray], None] = None,
        save: bool = True,
        identifier: str = "",
        index: Union[int, None] = None,
    ) -> tuple[Figure, Axes]:
        """Plot timeseries from model and observations."""

        cls.__clear__()
        cls.__setup_config__()

        n_series = len(ts_list)
        colors = ["k", "g", "b"]
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=cls.figsize_standard)

        loop = range(n_series)
        if index is not None:
            loop = [index]
        for i in loop:
            ts = ts_list[i]
            data = data_list[i].T

            kwargs = {}
            if legends is not None:
                kwargs = {"label": legends[i], "color": colors[i]}
            # if i == 0:
            #     kwargs |= {"color": "k", "zorder": 3}
            # else:
            #     kwargs |= {"alpha": cls.ensemble_alpha, "zorder": -1}
            if obs_data is not None:
                ax.plot(
                    obs_data[0],
                    obs_data[1],
                    "x",
                    label="Observation",
                    zorder=4,
                    color="r",
                )
                obs_data = None
            ax.plot(ts, data, **kwargs)

        ax.set_xlabel(cls.t_label)
        ax.set_ylabel(cls.h_label)
        ax.set_title(title_func(loc_name))

        if obs_data is not None or legends is not None:
            cls.legend(ax)
        cls.grid(ax)
        cls.date_axis(ax)

        if save:
            name = f"comp_{loc_name}_{identifier}.pdf".replace(" ", "_")
            plt.savefig(cls.add_folder(name), bbox_inches="tight")
        return fig, ax

    @classmethod
    def plot_series_locs(
        cls,
        ts: list,
        series_data: np.ndarray,
        loc_names: list[str],
        title_func: Callable,
        obs_data: Union[np.ndarray, None] = None,
        save: bool = True,
    ) -> None:
        """Plot timeseries from model and observations."""

        cls.__clear__()
        cls.__setup_config__()

        n_locs = len(loc_names)

        series_data = series_data.T
        for i in range(n_locs):
            location_label = loc_names[i]
            _, ax = plt.subplots(1, 1, figsize=cls.figsize_standard)
            ax.plot(ts, series_data[i, :], "b-", label="Simulation")

            if obs_data is not None:
                ax.plot(ts, obs_data, "k-", label="Observation")

            ax.set_xlabel(cls.t_label)
            ax.set_ylabel(cls.h_label)
            ax.set_title(title_func(location_label))

            if obs_data is not None:
                cls.legend(ax)
            cls.grid(ax)
            cls.date_axis(ax)

            if save:
                name = f"{location_label}.pdf".replace(" ", "_")
                plt.savefig(cls.add_folder(name), bbox_inches="tight")

    @classmethod
    def plot_map(
        cls,
        x,
        y,
        z,
        path: Union[str, None] = None,
        xlabel: str = "Longitude",
        ylabel: str = "Latitude",
        title: Union[str, None] = None,
        size_lims: tuple[float, float] = (0.001, 0.1),
        flip_lims: bool = False,
        size: Union[float, np.ndarray, list, None] = None,
        dpi: Union[int, None] = None,
        cmap: Union[str, None] = None,
        vmin: float | None = None,
        vmax: float | None = None,
        stations: Union[np.ndarray, None] = None,
        zorder_land: int = 15,
        draw_colorbar: bool = True,
        extent: tuple[float, float, float, float] | None = None,
    ) -> tuple[Figure, Axes]:
        """It creates a bathymetry plot with standard formatting.

        Parameters
        ----------
        x: Array-like
            Longitude coordinates.
        y: Array-like
            Latitude coordinates.
        z: Array-like
            The value at each coordinate.
        path: str | None, optional
            The name to save the figure with. Default: None
        xlabel: str, optional
            The label of the horizontal axis. Default: "$x$"
        ylabel: str, optional
            The label of the vertical axis. Default: "$y$"
        title: str | None, optional
            The title for the axes. Default: None
        size_limits: tuple[float, float], optional
            The limits to scale the scatter markers. Default: (0.001, 0.1)
        size: float | None, optional
            The marker size for the scatter points. Default: None
        dpi: int | None, optional
            The figure's DPI for saving. Default: None
        cmap: str | None, optional
            If a different colormap is required.
        vmin: float | None, optional
            The minimum value for the colorbar. Default: None
        vmax: float | None, optional
            The maximum value for the colorbar. Default: None
        stations: numpy.ndarray | None, optional
            Measuring stations to plot on the map. Array (lon, lat). Default: None
        draw_colorbar: bool, optional
            If the colorbar should be drawn. Default: True
        extent: tuple[float, float, float, float] | None, optional
            The GeoAxes extent. Default: None

        Returns
        -------
        matplotlib.figure.Figure
            The figure handle.
        matplotlib.figure.Axes
            The axes handle.
        """

        if dpi is None:
            dpi = cls.dpi
        cls.__clear__()
        cls.__setup_config__()

        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=cls.figsize_map,
            subplot_kw=dict(projection=cls.display_projection),
        )

        ax = cast(GeoAxes, ax)
        ax.set_extent(extent, crs=cls.orig_projection)
        if size is None:
            size = cls.get_sizes(z, size_lims)
            if flip_lims:
                size = size[::-1]

        if cmap is None:
            cmap = cls.color_map

        kwargs = {}
        if vmin is not None:
            kwargs |= {"vmin": vmin}
        if vmax is not None:
            kwargs |= {"vmax": vmax}
        c = ax.scatter(
            x,
            y,
            c=z,
            s=size,
            edgecolors="none",
            transform=cls.orig_projection,
            cmap=cmap,
            zorder=1,
            **kwargs,
        )

        extend = None
        if vmin is not None:
            extend = "min"
            if vmax is not None:
                extend = "both"
        if vmax is not None:
            extend = "max"

        kwargs = {"extend": extend}
        if draw_colorbar:
            plt.colorbar(c, **kwargs)  # type: ignore

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.coastlines(
            resolution=cls.coast_resolution, linewidth=cls.coast_width, zorder=10
        )
        ax.add_feature(cfeature.LAND, facecolor=cls.land_color, zorder=zorder_land)
        cls.map_grid(ax)

        if stations is not None:
            ax.scatter(
                stations[:, 0],
                stations[:, 1],
                c=cls.station_color,
                s=5,
                edgecolor="none",
                transform=cls.orig_projection,
                zorder=5,
            )

        if title is not None:
            ax.set_title(title)

        if path is not None:
            plt.savefig(cls.add_folder(path), bbox_inches="tight", dpi=dpi)
        return fig, ax

    @classmethod
    def compare_map(
        cls,
        xs,
        ys,
        zs: np.ndarray,
        path: Union[str, None] = None,
        xlabel: str = "Longitude",
        ylabel: str = "Latitude",
        titles: Union[list[str], None] = None,
        size_lims: tuple[float, float] = (0.001, 0.1),
        size: Union[float, np.ndarray, None] = None,
        dpi: Union[int, None] = None,
    ) -> tuple[Figure, np.ndarray]:
        """It creates a bathymetry plot with standard formatting.

        Parameters
        ----------
        xs: Array-like
            Longitude coordinates for each set of values.
        ys: Array-like
            Latitude coordinates for each set of values.
        zs: Array-like, shape (2, values)
            The two values to compare at each coordinate.
        path: str | None, optional
            The name to save the figure with. Default: None
        xlabel: str, optional
            The label of the horizontal axis. Default: "$x$"
        ylabel: str, optional
            The label of the vertical axis. Default: "$y$"
        titles: list[str] | None, optional
            The titles for each subplot. Default: None
        size_limits: tuple[float, float], optional
            The limits to scale the scatter markers. Default: (0.001, 0.1)
        size: float | None, optional
            The marker size for the scatter points. Default: None
        dpi: int | None, optional
            The figure's DPI for saving. Default: None

        Returns
        -------
        matplotlib.figure.Figure
            The figure handle.
        numpy.ndarray
            The axes handle array.
        """

        if dpi is None:
            dpi = cls.dpi
        cls.__clear__()
        cls.__setup_config__()

        fig, axs = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=cls.figsize_compare_map,
            subplot_kw=dict(projection=cls.display_projection),
        )
        z_min = zs.min()
        z_max = zs.max()
        for i in range(zs.shape[0]):
            ax = cast(GeoAxes, axs[i])
            x = xs[i, :]
            y = ys[i, :]
            z = zs[i, :]

            if size is None:
                size = cls.get_sizes(z, size_lims)

            c = ax.scatter(
                x,
                y,
                c=z,
                s=size,
                edgecolors="none",
                vmin=z_min,
                vmax=z_max,
                transform=cls.orig_projection,
                zorder=-1,
            )

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            ax.coastlines(resolution=cls.coast_resolution, linewidth=cls.coast_width)
            ax.add_feature(cfeature.LAND, facecolor=cls.land_color)
            cls.map_grid(ax)

            if titles is not None:
                ax.set_title(titles[i])

        fig.tight_layout()
        fig.subplots_adjust(right=0.94)
        cbar_ax = fig.add_axes([0.95, 0.1, 0.01, 0.8])
        fig.colorbar(c, cax=cbar_ax)  # type: ignore

        if path is not None:
            plt.savefig(cls.add_folder(path), bbox_inches="tight", dpi=dpi)
        return fig, axs

    @classmethod
    def plot_enkf(
        cls,
        ts: np.ndarray,
        ensembles: np.ndarray,
        observations: np.ndarray,
        station_name: str,
        method_label: str = "EnKF",
    ) -> None:
        """Plot the EnKF estimation over time.

        Parameters
        ----------
        ts: numpy.ndarray
            The time axis for the estimated states. Shape: (time,)
        ensembles: numpy.ndarray
            The matrix of ensemble estimations. (ensemble, time)
        observations: numpy.ndarray
            The matrix of observations. (2, obs_time) where the first row are the
            observation times and the second row are the observed values.
        """

        cls.__clear__()
        cls.__setup_config__()

        estimations = ensembles.mean(axis=0)
        stds = ensembles.std(axis=0)

        _, ax = plt.subplots(1, 1, figsize=cls.figsize_standard)
        ax.fill_between(
            ts,
            (estimations - stds),  # type: ignore
            (estimations + stds),  # type: ignore
            color="b",
            alpha=cls.bands_alpha,
            zorder=-1,
        )

        # Plot filter estimation
        ax.plot(ts, estimations, "b", label=method_label, zorder=3)

        # Plot actual observations
        ax.plot(
            observations[0, :],
            observations[1, :],
            "x",
            markevery=1,
            markersize=2,
            color="k",
            linewidth=1,
            label="Observations",
            zorder=2,
        )

        ax.set_xlabel(cls.t_label)
        ax.set_ylabel(cls.h_label)

        cls.grid(ax)
        cls.date_axis(ax)
        cls.legend(ax)

        name = f"{station_name}_series_{method_label}.pdf"
        plt.savefig(cls.add_folder(name), bbox_inches="tight")

    @classmethod
    def hist(
        cls,
        data: np.ndarray | list,
        bins: int | None,
        path: str | None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        normalize: bool = False,
    ) -> None:
        """It plots a histogram with standard formatting.

        Parameters
        ----------
        data: numpy.ndarray | list
            The data to create the histogram of.
        bins: int | None, optional
            The number of bins to use. Default: None
        path: str | None, optional
            The path to save the figure. Default: None
        xlabel: str | None, optional
            The label for the horizontal axis. Default: None
        ylabel: str | None, optional
            The label for the vertical axis. Default: None
        normalize: bool, optional
            If the histogram should be normalized (density). Default: False
        """

        cls.__clear__()
        cls.__setup_config__()

        _, ax = plt.subplots(1, 1, figsize=cls.figsize_standard)

        kwargs = {"color": "skyblue", "ec": "white", "lw": 0.3}
        if bins is not None:
            kwargs |= {"bins": bins}
        ax.hist(data, density=normalize, **kwargs)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        cls.grid(ax)

        if path is not None:
            plt.savefig(cls.add_folder(path), bbox_inches="tight", dpi=cls.dpi)
