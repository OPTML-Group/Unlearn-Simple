#!/bin/bash

set -e

echo "Starting unlearning model experiments..."

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/exec/unlearn_model.py \
    --config-file configs/unlearn/wmdp/NPO.json

echo "Finished: NPO+FT"

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/exec/unlearn_model.py \
    --config-file configs/unlearn/wmdp/SimNPO.json

echo "Finished: SimNPO+FT"

echo "All experiments completed successfully!"
