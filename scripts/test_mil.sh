#!/usr/bin/env bash

MIL_CONFIG=$1
MIL_MODEL_PATH=$2
GPUS=$3

bash MIL/tools/dist_test.sh $MIL_CONFIG $MIL_MODEL_PATH $GPUS --metrics accuracy