#!/bin/bash

source env_npu.sh

python3 run_generation.py \
    --model_type=bloom \
    --model_name_or_path=bloom-560m