#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -p genoa
#SBATCH -N 1

# load modules
module purge
module load 2022

export PYTHONEXE=~/.conda/envs/cartopy-dev/bin/python

$PYTHONEXE p2_preprocess_ERA5.py
wait
