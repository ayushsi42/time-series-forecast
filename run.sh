#!/bin/bash

# Hull Tactical Market Prediction Pipeline
# =========================================

# Default run with optimized settings
python workflow.py \
    --train-path train.csv \
    --test-path test.csv \
    --imputer-epochs 10 \
    --imputer-hidden 128 \
    --imputer-window 30 \
    --epochs 100 \
    --hidden-dim 512 \
    --seq-length 60 \
    --lr 0.0001 \
    --eval-every 10
