import numpy as np
from pathlib import Path
import pandas as pd
from netCDF4 import Dataset  # type: ignore
import xarray as xr
import click


def waterlevel2pressure(waterlevel, g=9.813, water_density=1023) -> float:
    return -1 * waterlevel * g * water_density


def get_pressure_array(time_array: np.ndarray, pressure_grid: np.ndarray) -> np.ndarray:
    """"""

    pressure_array = np.ones((time_array.shape[0], *pressure_grid.shape))
    return pressure_grid * pressure_array


def get_time_array(
    start_time: str, stop_time: str, time_incr: str, is_constant: bool = False
) -> np.ndarray:
    """"""

    start = pd.to_datetime(start_time, format="%Y%m%d_%H%M%S").to_numpy()  # type: ignore
    stop = pd.to_datetime(stop_time, format="%Y%m%d_%H%M%S").to_numpy()  # type: ignore
    incr = pd.to_timedelta(time_incr).to_numpy()

    if not is_constant:
        time = np.arange(start, stop + incr, incr, dtype="datetime64[ns]")
    else:
        time = np.array([start, stop], dtype="datetime64[ns]")
    return time


@click.command()
@click.option(
    "--start",
    help="Start Time, format YYYYMMDD_HHMMSS",
    default="20140525_000000",
)
@click.option(
    "--stop",
    help="Stop Time, format YYYY-MM-DDTHH:MM:SS",
    default="20140615_000000",
)
@click.option(
    "--incr",
    help="Time between increments, default='1D'",
    default="1D",
)
@click.option(
    "--bias",
    help="A waterlevel (m) bias to include in the pressure file.",
    default=0.0,
)
@click.option(
    "--output-file",
    help="The file to write the pressure file.",
    default="pressure.nc",
)
@click.option(
    "--variable",
    help="The pressure variable name to write on the netCDF file.",
    default="p",
)
@click.argument("input_mdt_file")
def main(
    start: str,
    stop: str,
    incr: str,
    bias: float,
    output_file: str,
    variable: str,
    input_mdt_file: str,
) -> None:
    """"""

    time_array = get_time_array(start, stop, incr, is_constant=True)
    # mdt = xr.load_dataset(input_mdt_file).isel(time=0).mdt
    mdt = xr.load_dataset(input_mdt_file).isel(time=0).fillna(0).mdt
    shifted_mdt = mdt + bias
    pressure_data = xr.apply_ufunc(
        waterlevel2pressure, shifted_mdt, vectorize=True
    ).values

    dataset = Dataset(output_file, mode="w", format="NETCDF4_CLASSIC")
    xs = mdt.longitude.values
    ys = mdt.latitude.values

    dataset.createDimension("time", size=time_array.shape[0])
    dataset.createDimension("x", size=xs.shape[0])
    dataset.createDimension("y", size=ys.shape[0])

    start_time = time_array[0]
    str_start_time = str(start_time.astype("datetime64[s]")).replace("T", " ")
    var_time = dataset.createVariable("time", "d", dimensions="time")
    var_time.setncattr("time_origin", str_start_time)
    var_time.setncattr(
        "long_name", "Time - minutes since " + str_start_time + " +00:00"
    )
    var_time.setncattr("standard_name", "time")
    var_time.setncattr("calender", "gregorian")
    var_time.setncattr("units", "minutes since " + str_start_time + " +00:00")
    var_time[:] = (time_array - start_time).astype("timedelta64[m]").astype("float32")

    var_x = dataset.createVariable("x", "double", dimensions="x")
    var_x.setncattr("standard_name", "longitude")
    var_x.setncattr("long_name", "longitude")
    var_x.setncattr("units", "degrees_east")
    var_x[:] = xs

    var_y = dataset.createVariable("y", "double", dimensions="y")
    var_y.setncattr("standard_name", "latitude")
    var_y.setncattr("long_name", "latitude")
    var_y.setncattr("units", "degrees_north")
    var_y[:] = ys

    var_id = dataset.createVariable(
        variable, "double", ["time", "y", "x"], fill_value=1e20
    )
    var_id.setncattr("coordinates", "Time y x")
    var_id.setncattr("long_name", "Atmospheric Pressure")
    var_id.setncattr("standard_name", variable)
    var_id.setncattr("units", "Pa")
    var_id[...] = pressure_data
    print(dataset.data_model)
    dataset.close()

    return dataset


if __name__ == "__main__":
    main()
