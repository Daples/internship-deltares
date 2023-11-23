#! /bin/bash
# start this script with sbatch <script name>
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --partition=genoa
#SBATCH --job-name=gtsm3_5km
#SBATCH --time=0-12:00:00 # time limit

#
#
# --- You will need to change the lines below -----------------------------

module purge
module load 2022 #on Snellius
module load intel/2022a
 
# Set number of partitions
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


# First: partitioning 
# (You can re-use a partition if the input files and the number of partitions haven't changed)
# Partitioning is executed by dflowfm, in the folder containing the mdu file
echo partitioning...
# "-p": See above. Arguments after "run_dflowfm.sh" are explained in run_dflowfm.sh
srun -n1 $singularitydir/execute_singularity_snellius.sh -p 1 run_dflowfm.sh --partition:ndomains=$nPart:icgsolver=6:contiguous=0  $mdufile > dimr_part.log

# Second: computation
echo "TIMINGS $(basename $PWD): Computation of $(basename $PWD), submitted at $(date) "
srun  --nodes 1 --exclusive -n $nPart $singularitydir/execute_singularity_snellius.sh -p 0 run_dflowfm.sh $mdufile > dimr.log
printf "TIMINGS $(basename $PWD): $(date) WAITING FOR SUBMIT COMMAND TO FINISH\n"
wait

#Calculate finish time
ELAPSED=$(($(date +%s) - START_TIME))
printf "TIMINGS $(basename $PWD): Computation of $(basename $PWD), completed at $(date), execution time $(date -d@$ELAPSED -u +%H\ hours\ %M\ min\ %S\ sec)\n"