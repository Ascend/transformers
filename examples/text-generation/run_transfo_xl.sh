#!/bin/bash

source env_npu.sh

python3 run_generation.py \
    --model_type=transfo-xl \
    --model_name_or_path=transfo-xl-wt103