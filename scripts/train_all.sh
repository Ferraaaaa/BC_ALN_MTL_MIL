#!/usr/bin/env bash

MTL_CONFIG=$1
MIL_CONFIG=$2
GPUS=$3

bash scripts/train_mtl.sh $MTL_CONFIG $GPUS

bash scripts/inference_mtl.sh $MTL_CONFIG

bash scripts/train_mil.sh $MIL_CONFIG $GPUS

bash scripts/inference_mil.sh $MIL_CONFIG