#!/bin/bash

$PYTHONEXE \
utils/xml_builder.py \
--locs-file \
stochModel/input_dflowfm/grid_locs.xyn \
--noos-folder \
noos_cmems \
-o \
stochObserver/noosObservationsAveraged.xml \
--template \
stochObserver/templates/noosObservations_template.xml \
make-noos-observer \
stochObserver/noos_cmems/timeseries_*

$PYTHONEXE \
utils/xml_builder.py \
--locs-file \
stochModel/input_dflowfm/grid_locs.xyn \
--noos-folder \
noos_cmems \
-o \
stochModel/dflowfmModel_TemporalAveraging.xml \
--template \
stochModel/templates/dflowfmModel_template.xml \
make-model \
--npart 20 \

$PYTHONEXE \
utils/xml_builder.py \
--locs-file \
stochModel/input_dflowfm/grid_locs.xyn \
--noos-folder \
noos_cmems \
-o \
stochModel/dflowfmStochModel_TemporalAveraging.xml \
--template \
stochModel/templates/dflowfmStochModel_template.xml \
make-stoch-model \
--npart 20 \
