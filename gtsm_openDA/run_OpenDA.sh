#!/bin/bash

module load 2022
module load ANTLR/2.7.7-GCCcore-11.3.0-Java-11

source $OPENDADIR/settings_local.sh linux

export ODA_JAVAOPTS=' '
# export OMP_NUM_THREADS=1

exec $OPENDADIR/oda_run.sh -o=openda_logfile.txt $1