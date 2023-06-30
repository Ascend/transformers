#!/bin/bash

source env_npu.sh

python3 run_generation.py \
    --model_type=opt \
    --model_name_or_path=opt-125m