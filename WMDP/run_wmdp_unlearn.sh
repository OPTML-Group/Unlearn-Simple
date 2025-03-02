#!/bin/bash

set -e

echo "Starting unlearning model experiments..."

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/exec/unlearn_model.py \
    --config-file configs/unlearn/wmdp/NPO.json \
    --unlearn.lr=5e-06 \
    --unlearn.NPO+FT.beta=0.0225 \
    --unlearn.NPO+FT.gamma=1.0 \
    --overall.seed=1001 \
    --unlearn.max_steps=125

echo "Finished: NPO+FT"

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/exec/unlearn_model.py \
    --config-file configs/unlearn/wmdp/SimNPO.json \
    --unlearn.lr=5e-06 \
    --unlearn.NPO+FT+SAM.beta=0.015 \
    --unlearn.NPO+FT+SAM.gamma=2.25 \
    --unlearn.NPO+FT+SAM.sam_rho=0.01 \
    --overall.seed=1001 \
    --unlearn.max_steps=125

echo "Finished: SimNPO+FT"

echo "All experiments completed successfully!"
