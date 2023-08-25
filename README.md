# Transformers
## 简介
:tada: Hugging Face 核心套件  [transformers](https://github.com/huggingface/transformers) 、  [accelerate](https://github.com/huggingface/accelerate) 、 [peft](https://github.com/huggingface/peft) 、  [trl](https://github.com/huggingface/trl)  已原生支持 Ascend NPU，相关功能陆续补齐中。

本仓用于发布最新进展、需求/问题跟踪、测试用例。


## 更新日志

[23/08/23] :tada: 现在 `transformers>=4.32.0` `accelearte>=0.22.0`、`peft>=0.5.0`、`trl>=0.5.0`原生支持 Ascend NPU！通过 `pip install` 即可安装体验:hugs:

[23/08/11] :sparkles:  [trl](https://github.com/huggingface/trl) 原生支持 Ascend NPU，请源码安装体验 :hugs:

[23/08/05] :sparkles: [accelerate](https://github.com/huggingface/accelerate) 支持 Ascend NPU 使用 FSDP(pt-2.1, Experimental)

[23/08/02] :sparkles:  [peft](https://github.com/huggingface/peft) 原生支持 Ascend NPU 的加载 adapter，请源码安装体验 :hugs:

[23/07/19] :sparkles:  [transformers](https://github.com/huggingface/transformers) 原生支持 Ascend NPU 的单卡/多卡/amp 训练，请源码安装体验 :hugs:

[23/07/12] :sparkles:  [accelerate](https://github.com/huggingface/accelerate) 原生支持 Ascend NPU 的单卡/多卡/amp 训练，请源码安装体验 :hugs:



## Transformers

### 使用说明

当前  [transformers](https://github.com/huggingface/transformers)  训练流程已原生支持 Ascend NPU，这里以参考示例中 [text-classification](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) 任务为例说明如何在 Ascend NPU 微调 bert 模型。

1.  请参考《[Pytorch框架训练环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fptes)》准备环境。

2.  安装 transformers

   ```
   pip3 install -U transformers
   ```

3. 获取[text-classification](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)训练脚本并安装相关依赖

   ```
   git clone https://github.com/huggingface/transformers.git
   cd examples/pytorch/text-classification
   pip install -r requirements.txt
   ```

4. 执行单卡训练

   参考 [text-classification](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) README，执行：

   ```
   export TASK_NAME=mrpc
   
   python run_glue.py \
     --model_name_or_path bert-base-cased \
     --task_name $TASK_NAME \
     --do_train \
     --do_eval \
     --max_seq_length 128 \
     --per_device_train_batch_size 32 \
     --learning_rate 2e-5 \
     --num_train_epochs 3 \
     --output_dir /tmp/$TASK_NAME/
   ```

   如果希望使用混合精度，请传入训练参数 `--fp16` 

   ```
   export TASK_NAME=mrpc
   
   python run_glue.py \
     --model_name_or_path bert-base-cased \
     --task_name $TASK_NAME \
     --do_train \
     --do_eval \
     --max_seq_length 128 \
     --per_device_train_batch_size 32 \
     --learning_rate 2e-5 \
     --num_train_epochs 3 \
     --fp16 \
     --output_dir /tmp/$TASK_NAME/
   ```

5. 执行多卡训练

   ```
   export TASK_NAME=mrpc
   
   python -m torch.distributed.launch --nproc_per_node=8 run_glue.py \
     --model_name_or_path bert-base-cased \
     --task_name $TASK_NAME \
     --do_train \
     --do_eval \
     --max_seq_length 128 \
     --per_device_train_batch_size 32 \
     --learning_rate 2e-5 \
     --num_train_epochs 3 \
     --output_dir /tmp/$TASK_NAME/
   ```

### 支持的特性

- [x] single NPU

- [x] multi-NPU on one node (machine)

- [x] FP16 mixed precision

- [x] PyTorch Fully Sharded Data Parallel (FSDP) support (Experimental)

  > need more test

- [ ] DeepSpeed support (Experimental)



## Accelerate

施工中...



## PEFT

施工中...



## TRL

施工中...



## FQA

施工中...
