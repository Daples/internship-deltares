#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --time=0-00:10:00
#SBATCH --partition=genoa
#SBATCH --exclusive
##SBATCH --contiguous

# Usage:
# D-Flow FM computations using a Singularity container,
# either sequential, or parallel computations using one node.
# For parallel using multiple nodes: use submit_singularity.sh.
#
# To start:
# 1. Be sure that a Singularity container is available, 
#    together with an execute_singularity.sh script in the same folder.
# 2. Copy the run_singularity script into your working folder, i.e. the folder containing the dimr config file.
# 3. Modify the run_singularity script, see remarks below.
# 4. Execute the script from the command line.
#    You can feed the script to a queueing system.
#
# "execute_singularity.sh -p 2": Parent level to mount:
# If your working folder does not contain all of the input files, then you must set the -p flag.
# Let's define the "top level" as the folder containing all of the input files.
# The value of -p must be the number of folder levels between the dimr config file and the top level.
# A higher value will not cause any harm, provided that folders exist at the higher levels.
# 

echo "Script submitted and running"
#
#
# --- You will need to change the lines below -----------------------------

#module load intelmpi/21.2.0 #on H6
module purge
module load 2022 #on Snellius
module load intel/2022a
 
# Set number of partitions (this script only works for one node)
nPart=20

# Set the path to the folder containing the singularity image
singularitydir=/home/mverlaan/einf220/fromDavid/delft3dfm_2023.02


mdufile=gtsm_fine.mdu

#
#
# --- You shouldn't need to change the lines below ------------------------

# stop after an error occurred:
set -e



#
#
# --- Execution part: modify if needed ------------------------------------

# Parallel computation on one node
#

# First: partitioning 
# (You can re-use a partition if the input files and the number of partitions haven't changed)

# Partitioning is executed by dflowfm, in the folder containing the mdu file
echo partitioning...
 # "-p": See above. Arguments after "run_dflowfm.sh" are explained in run_dflowfm.sh
srun -p genoa --time 01:00:00 -n 1 $singularitydir/execute_singularity_snellius.sh -p 0 run_dflowfm.sh --partition:ndomains=$nPart:icgsolver=6:contiguous=0  $mdufile