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
nohup python3 -m torch.distributed.run --nproc_per_node 8 run_speech_recognition_ctc.py \
    --dataset_name="common_voice" \
    --model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
    --dataset_config_name="tr" \
    --num_train_epochs="3" \
    --per_device_train_batch_size="4" \
    --learning_rate="3e-4" \
    --warmup_steps="500" \
    --evaluation_strategy="steps" \
    --text_column_name="sentence" \
    --length_column_name="input_length" \
    --save_steps="400" \
    --eval_steps="100" \
    --logging_steps="1" \
    --layerdrop="0.0" \
    --save_total_limit="3" \
    --freeze_feature_encoder \
    --gradient_checkpointing \
    --chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” � \
    --group_by_length \
    --do_train --do_eval \
    --overwrite_output_dir \
    --output_dir ./output > ${scripts_path_dir}/output/run_wav2vec2.log 2>&1 &
wait
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 打印端到端训练时间
echo "E2E Training Duration sec : $e2e_time"
