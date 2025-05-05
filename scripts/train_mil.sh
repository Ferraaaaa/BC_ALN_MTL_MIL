#!/usr/bin/env bash

MIL_CONFIG=$1
GPUS=$2

bash MIL/tools/dist_train.sh $MIL_CONFIG $GPUS