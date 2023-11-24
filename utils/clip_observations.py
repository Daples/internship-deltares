import xarray as xr
from utils.data_handler import DataHandler

obs_file = "./stochObserver/averaged_20cm.nc"
dest_file = "./stochObserver/clipped_averaged_20cm.nc"

obs_data = xr.load_dataset(obs_file)
clipped_data = DataHandler.clip_dataset(7, obs_data)
clipped_data.to_netcdf(dest_file)
