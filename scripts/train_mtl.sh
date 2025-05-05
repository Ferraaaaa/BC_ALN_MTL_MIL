#!/usr/bin/env bash

MTL_CONFIG=$1
GPUS=$2

bash MTL/tools/dist_train.sh $MTL_CONFIG $GPUS