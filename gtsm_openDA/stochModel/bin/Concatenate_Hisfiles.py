# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 16:39:40 2022

@author: zijlker
"""
import xarray as xr
import argparse
import os
from glob import glob


def Concatenate_Hisfiles(inputfiles_glob, outputfilename):
    # Generate list of filepaths
    # paths= glob.glob(inputfiles_glob)
    paths = inputfiles_glob
    # Load datasets and concatenate along time dinemsion
    ds = xr.concat(
        [xr.open_dataset(p) for p in paths],
        data_vars="minimal",
        dim="time",
        compat="override",
        coords="minimal",
    )

    # drop duplicate time entries, keep last as this contains analysis value
    ds = ds.drop_duplicates("time", keep="last")

    # write to netcdf
    ds.to_netcdf(outputfilename)


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_files", nargs="*", help="""Specify files to concatenate to concatenate"""
)
parser.add_argument("--output_filename", help=""""Specify output filename""")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args.input_files)
    print(args.output_filename)
    Concatenate_Hisfiles(args.input_files, args.output_filename)
    # As a final step: remove all individual hisfiles
    # for file in args.input_files:
    #     os.remove(file)
