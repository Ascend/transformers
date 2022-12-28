#!/bin/bash
cur_path=`pwd`
#集合通信参数,不需要修改
export RANK_SIZE=8
# 数据集路径,保持为空,不需要修改
data_path=""
device_id=0

#基础参数，需要模型审视修改
#训练模型
model_name=albert-base-v2
#训练epoch
train_epochs=2
#训练batch_size, 该参数外部可传入
batch_size=12
warmup_ratio=0.0
weight_decay=0.0

#获取外部传参，可扩展
for para in $*
do
    if [[ $para == --model_name* ]];then
      model_name=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
      batch_size=`echo ${para#*=}`
    elif [[ $para == --warmup_ratio* ]];then
      warmup_ratio=`echo ${para#*=}`
    elif [[ $para == --weight_decay* ]];then
      weight_decay=`echo ${para#*=}`
    fi
done

#网络名称，同目录名称
Network=${model_name}_for_PyTorch

###############指定训练脚本执行路径###############
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

if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi

#################创建日志输出目录，不需要修改#################
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

#训练开始时间，不需要修改
start_time=$(date +%s)
# 非平台场景时source 环境变量
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
fi

nohup python3.7 -m torch.distributed.launch --nproc_per_node 8 run_qa.py \
    --model_name_or_path ./${model_name} \
    --config_name ./${model_name}/config.json \
    --tokenizer_name ./${model_name} \
    --dataset_name squad \
    --do_train \
    --do_eval \
    --num_train_epochs ${train_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --max_eval_samples 2000 \
    --max_seq_length 348 \
    --pad_to_max_length \
    --doc_stride 128 \
    --fp16 \
    --fp16_opt_level O2 \
    --use_combine_grad \
    --optim adamw_apex_fused_npu \
    --dataloader_drop_last true \
    --warmup_ratio ${warmup_ratio} \
    --weight_decay ${weight_decay} \
    --overwrite_output_dir \
    --output_dir ./output > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

#结果打印，不需要修改
echo "------------------ Final result ------------------"
#输出性能FPS，需要模型审视修改
FPS=`grep "train_samples_per_second ="  ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $3}'`

#打印，不需要修改
echo "Final Performance images/sec : $FPS"

#输出训练精度,需要模型审视修改
train_accuracy=`grep "eval_f1" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | awk 'END {print $3}'`

#打印，不需要修改
echo "Final Train Accuracy : ${train_accuracy}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

##获取性能数据，不需要修改
#吞吐量
ActualFPS=${FPS}
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

#从train_$ASCEND_DEVICE_ID.log提取Loss到train_${CaseName}_loss.txt中，需要根据模型审视
grep "{'loss" ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log |awk -F "{'loss" '{print $2}'|awk '{print $2}' | awk -F "," '{print $1}' > ${test_path_dir}/output/${ASCEND_DEVICE_ID}//train_${CaseName}_loss.txt
#最后一个迭代loss值，不需要修改
ActualLoss=`awk 'END {print}' ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${CaseName}_loss.txt`

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "CaseName = ${CaseName}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log
echo "TrainAccuracy = ${train_accuracy}">> ${test_path_dir}/output/${ASCEND_DEVICE_ID}/${CaseName}.log