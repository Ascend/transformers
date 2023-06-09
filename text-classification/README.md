# Text classification examples

## 说明
本项目用于测试text-classification任务下transformers套件算法在昇腾NPU上的满足度。

## GLUE 任务

[`run_glue.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py)脚本用于训练sequence classification任务，GLUE由9个不同的子任务组成。下面是在其中一个子任务上运行的脚本：

```bash
export TASK_NAME=sst2

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --use_ascend \
  --npu_fp16 \
  --npu_fp16_opt_level O2 \
  --use_combine_grad \
  --output_dir /tmp/$TASK_NAME/
```
其中`task_name`可以是 `cola`、`sst2`、`mrpc`、`stsb`、`qqp`、`mnli`、`qnli`、`rte`、`wnli`。

## 准备训练环境
### 准备环境
- 当前任务支持的PyTorch版本和已知三方库依赖表如下所示。

    **表 1** 版本支持表

| Torch_Vserion | 三方库依赖版本   |
|---------------|-----------|
| PyTorch 1.8   | apex==0.1 |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。
  1. 安装 optimum-ascend 插件
  ```bash
  git clone -b develop https://gitee.com/ascend/transformers.git
  cd transformers
  pip3 install -e ./
  ```
  2. 安装text-classification的依赖
  ```bash
  pip3 install -r requirements.txt
  ```

### 准备预训练模型
请从[huggingface官网](https://huggingface.co/models)获取对应模型的预训练权重，将下载好的预训练权重上传至与本项目同级的`model_path`目录下。
预训练权重目录结构参考如下所示。
```text
   $ model_path
       ├── bert-base-cased
       │   ├── README.md
       │   ├── pytorch_model.bin
       │   ├── tokenizer.json
       │   ├── tokenizer_config.json
       │   └── vocab.txt
       │           
       └── ... 
```
> 说明： 该预训练权重只作为一种参考示例，不同预训练权重包含的内容由些许差异。

## 开始训练
### 训练模型
1. 进入`text-classification` 目录。
   ```bash
   cd ./text-classification
   ```
2. 运行训练脚本
   ```bash
   bash ./test/run_bert_base_cased.sh
   ```

### 训练结果
   
## 附
### 测试算法及预训练权重链接
逐步补充