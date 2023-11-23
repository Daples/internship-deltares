import numpy as np
from pathlib import Path
import pandas as pd
import argparse
from pprint import pprint
import os

divide = "\r" + "".join(["*"] * 25) + "\n"


def Create_TimeArray(start_time, stop_time, time_incr):
    start_time = pd.to_datetime(start_time, format="%Y%m%d_%H%M%S").to_numpy()
    stop_time = pd.to_datetime(stop_time, format="%Y%m%d_%H%M%S").to_numpy()
    time_incr = pd.to_timedelta(time_incr).to_numpy()
    return np.arange(
        start_time, stop_time + time_incr, time_incr, dtype="datetime64[ns]"
    )


def Create_Pmsl(gConst=9.813, rhow=1023, bias=0.0):
    # This bias changes the bias in m to pressure (Pa)
    # gConst is the gravitational constant
    # rhow is the density of water
    # bias is in meters [m]
    return -1 * bias * gConst * rhow


def Create_NewGrid(args):
    xs = np.linspace(
        args.x_llcenter, args.x_llcenter + args.dx * args.n_cols - 1, num=args.n_cols
    )
    ys = np.linspace(
        args.y_llcenter, args.y_llcenter + args.dy * args.n_rows - 1, num=args.n_rows
    )
    return xs, ys


def Create_Uniform_PressureData(time_array, xs, ys, Pmsl):
    Pressuredata = np.ones((len(time_array), len(ys), len(xs))) * Pmsl
    return Pressuredata


def Create_AMP_FILE(time_array, args, PressureData, filename):
    header = f"""### START OF HEADER
### This file is created by Deltares
### Additional comments
FileVersion = 1.03
filetype = meteo_on_equidistant_grid
NODATA_value = -999.0
n_cols = {args.n_cols}
n_rows = {args.n_rows}
grid_unit = degree
x_llcenter = {args.x_llcenter}
y_llcenter = {args.y_llcenter}
dx = {args.dx}
dy = {args.dy}              
n_quantity = {args.n_quantity}
quantity1 = {args.quantity1}
unit1 = {args.unit1}
### END OF HEADER\n"""

    start_time = time_array[0]
    str_start_time = str(start_time.astype("datetime64[s]")).replace("T", " ")

    hrs_after_CE = (time_array - start_time).astype("timedelta64[h]")
    print("Write to ", filename)
    f = open(filename, "w")
    f.write(header)
    for i, hr_since_CE in enumerate(hrs_after_CE):
        one = f"""TIME = {hr_since_CE} since {str_start_time} +00:00\n"""
        f.write(one)
        f.write(
            "\n".join(
                [
                    " ".join(map(lambda x: "{:.2f}".format(x), row))
                    for row in PressureData[i, ...]
                ]
            )
        )
        f.write("\n")
    f.close()
    print("Close AMP File")
    print(divide)
    return f


def Create_NC_FILE(time_array, xs, ys, PressureData, nc_format, args, filename):
    from netCDF4 import Dataset  # type: ignore

    # Pressure_data = Pmsl * np.ones([time_array.shape[0], ys.shape[0], xs.shape[0]], dtype='float64')
    print("Write to ", filename)
    rootgrp = Dataset(filename, mode="w", format=nc_format)

    dim_time = rootgrp.createDimension("time", size=time_array.shape[0])
    dim_x = rootgrp.createDimension("x", size=xs.shape[0])
    dim_y = rootgrp.createDimension("y", size=ys.shape[0])

    start_time = time_array[0]
    str_start_time = str(start_time.astype("datetime64[s]")).replace("T", " ")
    var_time = rootgrp.createVariable("time", "d", dimensions="time")
    var_time.setncattr("time_origin", str_start_time)
    var_time.setncattr(
        "long_name", "Time - minutes since " + str_start_time + " +00:00"
    )
    var_time.setncattr("standard_name", "time")
    var_time.setncattr("calender", "gregorian")
    var_time.setncattr("units", "minutes since " + str_start_time + " +00:00")
    var_time[:] = (time_array - start_time).astype("timedelta64[m]").astype("float32")

    var_x = rootgrp.createVariable("x", "double", dimensions="x")
    var_x.setncattr("standard_name", "longitude")
    var_x.setncattr("long_name", "longitude")
    var_x.setncattr("units", "degrees_east")
    var_x[:] = xs

    var_y = rootgrp.createVariable("y", "double", dimensions="y")
    var_y.setncattr("standard_name", "latitude")
    var_y.setncattr("long_name", "latitude")
    var_y.setncattr("units", "degrees_north")
    var_y[:] = np.flip(ys)  # Necessary to cast netcdf in the right format!

    var_id = rootgrp.createVariable(args.quantity1, "double", ["time", "y", "x"])
    var_id.setncattr("coordinates", "Time y x")
    var_id.setncattr("long_name", "Atmospheric Pressure")
    var_id.setncattr("standard_name", args.quantity1)
    var_id.setncattr("units", "Pa")
    var_id[...] = PressureData
    print(rootgrp.data_model)

    rootgrp.close()
    print("Close NetCDF File")
    print(divide)
    return rootgrp


def BiasConfigurationText(args):
    from datetime import datetime

    now = datetime.now()
    text = [divide]

    text.append("Noise Files Created at {} \n".format(now))
    text.append("Time Settings:\r")
    text.append("\tStart: {} \r".format(args.start))
    text.append("\tStop: {} \r".format(args.stop))
    text.append("\tIncrement: {} \r".format(args.incr))

    text.append("Bias settings:\r")
    text.append("\tFill_Value: {} \r".format(args.fill_value))
    text.append("\tQuantity name: {} \r".format(args.quantity1))
    text.append("\tQuantity unit: {} \r".format(args.unit1))

    text.append("Grid settings\r")
    text.append("\tNumber of columns {} \r".format(args.n_cols))
    text.append("\tNumber of rows: {} \r".format(args.n_rows))
    text.append("\tdx: {} \r".format(args.dx))
    text.append("\tdy: {} \r".format(args.dy))
    text.append("\tLower left cell x: {} \r".format(args.x_llcenter))
    text.append("\tLower left cell y: {} \r".format(args.y_llcenter))
    text.append("\tGrid units: {} \r".format(args.grid_unit))

    text.append("Save Settings:\r")
    text.append("\tFolder: {} \r".format(args.folder))
    text.append("\tFile Name: {} \r".format(args.filename))
    text.append("\tExtension: {} \r".format(args.ext))
    text.append("\t(Optional NetCDF Extension: {} \r".format(args.nc_format))

    text.append(divide)
    return text


def main(args):
    if not os.path.exists(Path(args.folder)):
        os.mkdir(Path(args.folder))

    # Array containing time
    time_array = Create_TimeArray(
        start_time=args.start, stop_time=args.stop, time_incr=args.incr
    )

    # Pressure value to fill the file
    Pmsl = Create_Pmsl(gConst=9.813, rhow=1023, bias=args.fill_value)

    # Grid value
    xs, ys = Create_NewGrid(args)

    # Array containing pressure data
    PressureData = Create_Uniform_PressureData(time_array, xs, ys, Pmsl)

    # Output text to write
    outputtext = BiasConfigurationText(args)

    # Output to Terminal
    list(map(print, outputtext))

    # Output to CurrentBiasConfiguration.txt
    f = open(Path(args.folder) / "BiasConfiguration.txt", "w")
    f.writelines([o.replace("\n", "\n\n").replace("\r", "\n") for o in outputtext])
    f.close()

    if "amp" in args.ext:
        # Create Air Pressure Noise AMP File
        amp_file = Create_AMP_FILE(
            time_array,
            args,
            PressureData,
            filename=Path(args.folder) / (args.filename + ".amp"),
        )

    if "netcdf" in args.ext:
        # Create Air Pressure Noise NC File
        nc_file = Create_NC_FILE(
            time_array,
            xs,
            ys,
            PressureData,
            args=args,
            filename=Path(args.folder) / (args.filename + ".nc"),
            nc_format=args.nc_format,
        )
    return 1


parser = argparse.ArgumentParser()
time = parser.add_argument_group(
    "Time", "Specify Time Settings of Pressure Noise Time series, i.e. "
)
time.add_argument(
    "--start",
    help="Start Time, format YYYYMMDD_HHMMSS, default='20130101_000000'",
    default="20130101_000000",
)
time.add_argument(
    "--stop",
    help="Stop Time, format YYYY-MM-DDTHH:MM:SS, default='20130110_000000'",
    default="20130101_000000",
)
time.add_argument("--incr", help="Time between increments, default='1D'", default="1D")

bias_ = parser.add_argument_group(
    "Bias", "Assign to the Pressure Noise Time series a non-time varying Bias"
)
bias_.add_argument(
    "--fill_value",
    help="Constant Bias across domain, default=0.0",
    type=float,
    default=0.0,
)
bias_.add_argument(
    "--n_quantity", type=int, help="Specify number of quantities in AMP file", default=1
)
bias_.add_argument(
    "--quantity1",
    type=str,
    help="Specify name of airpressure variable",
    default="air_pressure",
)
bias_.add_argument(
    "--unit1", type=str, help="Specify unit of airpressure variable", default="Pa"
)

grid = parser.add_argument_group(
    "Grid", "Specify Grid Settings of Pressure Noise file."
)
grid.add_argument("--n_cols", type=int, help="Specify number of columns", default=27)
grid.add_argument("--n_rows", type=int, help="Specify number of rows", default=16)
grid.add_argument("--grid_unit", help="Secify grid units", default="degree")
grid.add_argument(
    "--x_llcenter",
    type=float,
    help="Specify lower left cell center x-value",
    default=-12,
)
grid.add_argument(
    "--y_llcenter",
    type=float,
    help="Specify lower left cell center t-value",
    default=48,
)
grid.add_argument("--dx", type=float, help="Specify dx of grid", default=1)
grid.add_argument("--dy", type=float, help="Specify dy of grid", default=1)

file = parser.add_argument_group(
    "Save", "Pressure Noise Time series file Save Settings"
)
file.add_argument(
    "--folder",
    help="Specify Folder location, default='/home/oharak/'",
    default="/home/oharak/",
)
file.add_argument(
    "--filename",
    help="Specify FileName NOT INCL. EXTENSION!, the extension is appended, default='dcsmv5_airpressure_noise'",
    default="dcsmv5_airpressure_noise",
)
file.add_argument(
    "--ext", help="File Extension,", choices=["amp", "netcdf"], nargs="+", default="amp"
)
file.add_argument(
    "-nc_format",
    help="Specify the format of the NetCDF file to write",
    choices=["NETCDF3_CLASSIC", "NETCDF4_CLASSIC"],
    default="NETCDF4_CLASSIC",
)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
