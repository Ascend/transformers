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

#创建DeviceID输出目录，不需要修改
if [ -d ${test_path_dir}/output ];then
    rm -rf ${test_path_dir}/output
    mkdir -p ${test_path_dir}/output
else
    mkdir -p ${test_path_dir}/output
fi

################启动训练脚本################
start_time=$(date +%s)
nohup python3 -m torch.distributed.run --nproc_per_node 8 run_mim.py \
  --config_name_or_path swin \
  --model_type swin \
  --overwrite_output_dir \
  --remove_unused_columns False \
  --label_names bool_masked_pos \
  --do_train \
  --do_eval \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --logging_strategy steps \
  --logging_steps 10 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end True \
  --save_total_limit 3 \
  --seed 1337 \
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
