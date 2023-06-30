#!/bin/bash

source env_npu.sh

python3 run_generation.py \
    --model_type=llama \
    --model_name_or_path=llama-7b-hf