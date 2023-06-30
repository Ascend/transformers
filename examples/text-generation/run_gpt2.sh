#!/bin/bash

source env_npu.sh

python3 run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=gpt2