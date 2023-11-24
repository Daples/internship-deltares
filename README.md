# Application of Bias EnKF for the Global Tide and Surge Model (GTSM)

This repository contains most of my work during my internship at Deltares, from August
2023 to November 2023, under the supervision of Martin Verlaan and Tammo Zijlker.

The folder `notebooks` contains the notebooks used to obtain the results based on
simulations of [GTSM](https://publicwiki.deltares.nl/display/GTSM/Global+Tide+and+Surge+Model),
as well as the assimilation experiments and data processing.

The folder `utils` has the code for data processing utilities, OpenDA XML creation,
gridding of CMEMs data, and visualization tools.

The folder `gtsm_openDA` contains the template to setup OpenDA with GTSM in
[Snellius](https://www.surf.nl/en/dutch-national-supercomputer-snellius). It contains
all the scripts to generate the pressure and MDT forcing, running jobon Snellius,
extracting observation files (`.noos`) from netCDF data files and all the templates for
OpenDA XML files. GTSM should be included within `stochModel/input_dflowfm`.

This project requires `python3.11`, along with the libraries specified in
`requirements.txt`. GTSM requires [Delft3D FM](https://www.deltares.nl/en/software-and-data/products/delft3d-flexible-mesh-suite)
to run.
