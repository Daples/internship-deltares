#!/bin/bash
#SBATCH -t 8:00:00
#SBATCH -p genoa
#SBATCH -N 1

# load modules
module purge
module load 2022

export PYTHONEXE=~/.conda/envs/cartopy-dev/bin/python

# loop over months and years
for yr in {2014..2014..1}; do
  for mnth in {6..8..1}; do
  (
    echo $yr $mnth
    $PYTHONEXE p1a_download_ERA5.py $yr $mnth
  ) &
  done
done
wait
