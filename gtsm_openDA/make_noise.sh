$PYTHONEXE \
make_pressure_noise.py \
--start 20140525_000000 \
--stop 20140831_000000 \
--incr 1H \
--fill_value 0 \
--n_quantity 1 \
--quantity1 p \
--unit1 Pa \
--n_cols 361 \
--n_rows 181 \
--grid_unit degree \
--x_llcenter -180.0 \
--y_llcenter -90 \
--dx 1 \
--dy 1 \
--folder ./stochModel/input_dflowfm \
--filename airpressure_noise \
--ext netcdf \
-nc_format NETCDF3_CLASSIC