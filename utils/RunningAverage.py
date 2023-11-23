# -*- coding: utf-8 -*-
"""
@ Author Tammo Zijlker
"""
# Imports
import xarray as xr
import numpy as np
import argparse
import os
import datetime


def Rolling_Average_His_File(args):
    # Load dataset
    His = xr.load_dataset(args.his_filename_input)

    # Remove all irrelevant variables
    drop_vars_list = [
        var
        for var in list(His.data_vars)
        if not var
        in [
            args.variable,
            "wgs84",
            "station_id",
            "station_geom",
            "station_geom_node_count",
            "station_geom_node_coordx",
            "station_geom_node_coordy",
        ]
    ]
    His = His.drop_vars(drop_vars_list)

    # Find timestep in Hisfile
    His_timedelta = His["waterlevel"].time.values[1] - His["waterlevel"].time.values[0]

    # Calculate windowsize
    Windowsize = int(
        np.timedelta64(args.Rolling_Window_Time, args.Rolling_Window_Unit)
        / His_timedelta
    )

    # Apply moving average to hisfile
    His[args.variable] = (
        His[args.variable].rolling(dim={"time": Windowsize}, center=True).mean()
    )

    # Drop nan
    His = His.dropna(dim="time", how="all")

    # Save only the measurement at 12 o'clock
    His = His.sel(time=datetime.time(12))

    # Re-add dimension time
    try:
        His[args.variable] = His[args.variable].expand_dims("time")
    except:
        pass

    # To NetCDF
    His.to_netcdf(args.his_filename_output, format="NETCDF3_CLASSIC")

    print("Averaged hisfile written to : {}".format(os.getcwd()))
    return


parser = argparse.ArgumentParser(
    "Read Hisfile, apply a moving average window and write new hisfile"
)
parser.add_argument(
    "--his_filename_input",
    help="Path to His-file where a rolling average window should be applied",
)
parser.add_argument(
    "--his_filename_output",
    help="Path to where averaged His-file should be written away",
)
parser.add_argument(
    "--variable", type=str, help="Variable to which time averaging should be applied"
)
parser.add_argument(
    "--Rolling_Window_Time",
    type=int,
    help="Length of the averaging window to be applied (default = 10)",
    default=10,
)
parser.add_argument(
    "--Rolling_Window_Unit",
    type=str,
    help="Unit of the averaging window to be applied (for example 'D', 'm', 's' ), default = 's'",
    default="s",
)

if __name__ == "__main__":
    args = parser.parse_args()
    Rolling_Average_His_File(args)
