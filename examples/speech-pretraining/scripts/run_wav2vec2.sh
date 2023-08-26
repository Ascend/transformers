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
nohup accelerate launch -m torch.distributed.run --nproc_per_node 8 run_wav2vec2_pretraining_no_trainer.py \
  --dataset_name="librispeech_asr" \
	--dataset_config_names clean clean \
	--dataset_split_names validation test \
	--model_name_or_path="patrickvonplaten/wav2vec2-base-v2" \
	--max_train_steps="20000" \
	--num_warmup_steps="32000" \
	--gradient_accumulation_steps="8" \
	--learning_rate="0.005" \
	--weight_decay="0.01" \
	--max_duration_in_seconds="20.0" \
	--min_duration_in_seconds="2.0" \
	--logging_steps="1" \
	--saving_steps="10000" \
	--per_device_train_batch_size="8" \
	--per_device_eval_batch_size="8" \
	--adam_beta1="0.9" \
	--adam_beta2="0.98" \
	--adam_epsilon="1e-06" \
	--gradient_checkpointing \
	--mask_time_prob="0.65" \
	--mask_time_length="10" \
  --overwrite_output_dir \
  --output_dir ./output > ${scripts_path_dir}/output/run_wav2vec2.log 2>&1 &
wait
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 打印端到端训练时间
echo "E2E Training Duration sec : $e2e_time"
