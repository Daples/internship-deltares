import os
import xarray as xr
import dfm_tools as dfmt
import pandas as pd
from pandas.core.frame import DataFrame
from utils.data_handler import DataHandler


def read_noosfile(
    file_noos, datetime_format="%Y%m%d%H%M%S", na_values=None
) -> tuple[DataFrame, dict]:
    noosheader = []
    noosheader_dict = {}
    with open(file_noos) as f:
        startdata = 0
        for linenum, line in enumerate(f, 0):
            if "#" in line:
                noosheader.append(line)
                comment_stripped = line.strip("#").strip().split(": ")
                if len(comment_stripped) == 1:
                    if comment_stripped[0] != "":
                        noosheader_dict[comment_stripped[0]] = ""
                else:
                    noosheader_dict[comment_stripped[0].strip()] = comment_stripped[
                        1
                    ].strip()
            else:
                startdata = linenum
                break

    content_pd = pd.read_csv(
        file_noos,
        header=startdata - 1,
        delim_whitespace=True,
        names=["times_str", "values"],
        na_values=na_values,  # type: ignore
    )
    noos_datetime = pd.to_datetime(content_pd["times_str"], format=datetime_format)
    noosdata_pd = pd.DataFrame(
        {"datetime": noos_datetime, "values": content_pd["values"]}
    )

    return noosdata_pd, noosheader_dict


def write_noosfile(
    filename, pd_data, metadata=None, na_values=None, float_format="%.5f"
) -> None:
    import pandas as pd

    # import numpy as np

    with open("%s" % filename, "w") as file_noos:
        pass

    with open("%s" % filename, "a") as file_noos:
        file_noos.write(
            "# ----------------------------------------------------------------\n"
        )
        if metadata is not None:
            if type(metadata) == pd.DataFrame:
                col_headers = metadata.columns.tolist()
            elif type(metadata) == dict:
                col_headers = list(metadata.keys())
            else:
                raise Exception("metadata should be of type dict or pandas.DataFrame")
            # col_headerswidth = np.max([len(x) for x in col_headers])
            for iC, pdcol in enumerate(col_headers):
                if metadata[pdcol] == "":
                    file_noos.write("# {}\n".format(pdcol))
                else:
                    file_noos.write("# {:30} : {}\n".format(pdcol, metadata[pdcol]))
        if na_values is None:
            file_noos.write("# {:30} : {}\n".format("missing values", ""))
        else:
            file_noos.write("# {:30} : {}\n".format("missing values", na_values))
        file_noos.write(
            "# ----------------------------------------------------------------\n"
        )

    pd_data.to_csv(
        "%s" % filename,
        header=None,
        index=None,
        sep="\t",
        mode="a",
        date_format="%Y%m%d%H%M%S",
        float_format=float_format,
        na_rep=na_values,
    )


# Stations
xs, ys, stations = DataHandler.read_xyn("./stochModel/input_dflowfm/grid_locs_bk.xyn")

nc = xr.open_dataset("./stochObserver/sealevel_copernicus.nc")

# Stations to process

noos_folder = (
    "/home/mverlaan/einf220/fromDavid/gtsm_openDA_david/stochObserver/noos_cmems"
)
template_file = "/home/mverlaan/einf220/fromDavid/gtsm_openDA_david/stochObserver/templates/template.noos"
data_file = "/home/mverlaan/einf220/fromDavid/gtsm_openDA_david/stochObserver/sealevel_copernicus.nc"

# In and output locations of noosfiles
_, template_header = read_noosfile(template_file)

# Load simulation results
nc = xr.open_mfdataset(data_file)

# Write noosfiles
for x, y, station in zip(xs, ys, stations):
    noosfile_header_out = template_header.copy()

    # Customize header
    keys = list(noosfile_header_out.keys())
    noosfile_header_out[keys[4]] = station
    noosfile_header_out[keys[5]] = str((x, y))

    # ADT
    df = nc.adt.interp(longitude=x, latitude=y).to_series().dropna()

    # Edit dates
    start = "2014-06-01"
    stop = "2014-08-31"
    df = df[start:stop]
    df = df.reset_index()
    noosfile_header_out["Source"] = f"from {data_file}, daily average waterlevel"
    noosfilename_out = os.path.join(
        noos_folder, f"timeseries_{start}_{stop}_{station}_averaged.noos"
    )
    write_noosfile(
        noosfilename_out,
        df,
        metadata=noosfile_header_out,
    )
