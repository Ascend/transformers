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
nohup python3 -m torch.distributed.run --nproc_per_node 8 run_swag.py \
  --model_name_or_path distilbert-base-uncased \
  --do_train \
  --do_eval \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --per_gpu_eval_batch_size=16 \
  --per_device_train_batch_size=16 \
  --fp16 \
  --output_dir ./output > ${test_path_dir}/output/train.log 2>&1 &
wait

################获取训练数据################
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "E2E Training Duration sec : $e2e_time"
