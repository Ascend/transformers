# Text classification

本项目展示如何在Ascend NPU下运行Tansformers的[text-classification](https://github.com/huggingface/transformers/tree/v4.30.2/examples/pytorch/text-classification)任务。

## 准备训练环境
### 准备环境
- 环境准备指导。
  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  当前基于 PyTorch 1.11 完成测试。
- 安装依赖
  
  1、使用 NPU 设备源码安装适配昇腾的 accelerate
  ```text
  git clone -b accelerate-v0.21.0 https://gitee.com/ascend/transformers.git accelerate
  cd accelerate
  pip3 install -e .
  ```
  2、使用 NPU 设备源码安装 Transformers 插件
  ```text
  git clone -b v4.30.2 https://gitee.com/ascend/transformers.git
  cd transformers
  pip3 install -e .
  ```
  > 注：该插件依赖Transformers-v4.30.2，将会自动安装该版本

  3、安装text-classification任务所需依赖
  ```text
  pip3 install -r requirements.txt
  ```

### 准备数据集
text-classification示例使用的数据集由Hugging Face提供的接口自动下载无需额外准备

### 准备预训练权重
官方[text-classification](https://github.com/huggingface/transformers/tree/v4.30.2/examples/pytorch/text-classification)中`run_glue.py`和`run_glue_no_trainer.py`使用的预训练权重为[bert-base-cased](https://huggingface.co/bert-base-cased)，
`run_xnli.py`使用的预训练权重为[bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)，请按需下载相应的权重并置于`run_glue.py`同级目录下。

## 开始训练
- 使用Trainer类的用例
  ```text
  bash ./test/run_glue.sh
  ```
- 不使用Trainer类通过accelerate启动的用例
  ```text
  bash ./test/run_glue_no_trainer.sh
  ```
- 在 XNLI 数据集上微调
 ```text
 bash ./test/run_xnli.sh
 ```
> 注：run_glue.sh和run_glue_no_trainer.sh使用的数据集为sst2

## 训练结果
- `run_glue.sh` 使用单机8卡在910A上端到端训练耗时 8min，准确率(Acc)为 0.9163。
- `run_glue_no_trainer.sh` 使用单机8卡在910A上端到端训练耗时约 8min，准确率（Acc）为 0.913。

## 版本说明
### 变更
- 2023.06.26：Transformers版本更新到v4.30.2
- 2023.03.05: 首次发布

