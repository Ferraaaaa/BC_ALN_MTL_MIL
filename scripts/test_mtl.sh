#!/usr/bin/env bash

MTL_CONFIG=$1
MTL_MODEL_PATH=$2
GPUS=$3

bash MTL/tools/dist_test.sh $MTL_CONFIG $MTL_MODEL_PATH $GPUS --eval mIoU