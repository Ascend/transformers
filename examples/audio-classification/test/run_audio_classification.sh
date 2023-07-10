#!/bin/bash

# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

#创建输出目录，不需要修改
if [ -d ${test_path_dir}/output ];then
    rm -rf ${test_path_dir}/output
    mkdir -p ${test_path_dir}/output
else
    mkdir -p ${test_path_dir}/output
fi

################启动训练脚本################
start_time=$(date +%s)
# facebook/wav2vec2-base
nohup python3 -m torch.distributed.run --nproc_per_node 8 run_audio_classification.py \
  --model_name_or_path wav2vec2-base \
  --dataset_name superb \
  --dataset_config_name ks \
  --overwrite_output_dir \
  --remove_unused_columns False \
  --do_train \
  --do_eval \
  --learning_rate 3e-5 \
  --max_length_seconds 1 \
  --attention_mask False \
  --warmup_ratio 0.1 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 4 \
  --per_device_eval_batch_size 32 \
  --dataloader_num_workers 4 \
  --logging_strategy steps \
  --logging_steps 10 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end True \
  --metric_for_best_model accuracy \
  --save_total_limit 3 \
  --seed 0 \
  --fp16 \
  --fp16_opt_level O1 \
  --half_precision_backend apex \
  --loss_scale 16 \
  --output_dir ./output > ${test_path_dir}/output/train.log 2>&1 &
wait

################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"
