#!/bin/bash

source env_npu.sh

python3 run_generation.py \
    --model_type=xlm \
    --model_name_or_path=xlm-mlm-en-2048