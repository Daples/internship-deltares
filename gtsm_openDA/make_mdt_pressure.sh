$PYTHONEXE \
mdt2pressure.py \
--start 20140525_000000 \
--stop 20140831_000000 \
--incr 1H \
--bias 0 \
--output-file ./stochModel/input_dflowfm/climate_forcing/pressure_mdt.nc \
--variable p \
./stochModel/input_dflowfm/climate_forcing/mdt.nc