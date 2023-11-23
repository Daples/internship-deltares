import xarray as xr
from utils.data_handler import DataHandler

nc = xr.load_dataset("./stochModel/input_dflowfm/output/gtsm_fine_0000_his.nc")
nc_clipped = DataHandler.clip_dataset(1008, nc)
nc_clipped.to_netcdf("./stochObserver/clipped_data.nc")
