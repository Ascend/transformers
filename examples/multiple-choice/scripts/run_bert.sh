#!/bin/bash

cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"scripts" ];then
    scripts_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    scripts_path_dir=${cur_path}/scripts
fi

#创建输出目录，不需要修改
if [ -d ${scripts_path_dir}/output ];then
    rm -rf ${scripts_path_dir}/output
    mkdir -p ${scripts_path_dir}/output
else
    mkdir -p ${scripts_path_dir}/output
fi

# 启动训练脚本
start_time=$(date +%s)
nohup python3 -m torch.distributed.run --nproc_per_node 8 run_swag.py \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --per_device_eval_batch_size=16 \
  --per_device_train_batch_size=16 \
  --overwrite_output \
  --output_dir ./output > ${scripts_path_dir}/output/run_bert.log 2>&1 &
wait
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 打印端到端训练时间
echo "E2E Training Duration sec : $e2e_time"
