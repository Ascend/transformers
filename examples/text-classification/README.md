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
执行
```bash
bash ./test/run_xxx.sh
```

## 训练结果
- `run_glue.sh` 使用单机8卡在910A上端到端训练耗时 8min，准确率(Acc)为 0.9163。
- `run_glue_no_trainer.sh` 使用单机8卡在910A上端到端训练耗时约 8min，准确率（Acc）为 0.913。

| Architecture    | Pretrained Model                                                                                        | Script                                                                                                                           | Device | Performance(8-cards) | Accuracy |
|-----------------|---------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|--------|----------------------|----------|
| albert          | [albert-base-v2](https://huggingface.co/albert-base-v2)                                                 | [run_albert.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-classification/run_albert.sh)                   | 910B1  | 1438                 | 0.922    |
| bart            | [bart-base](https://huggingface.co/facebook/bart-base)                                                  | [run_bart.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-classification/run_bart.sh)                       | 910B1  | 667                  | 0.9312   |
| barthez         | [barthez](https://huggingface.co/moussaKam/barthez)                                                     | [run_barthez.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-classification/run_barthez.sh)                 | 910B1  | 637                  | 0.8601   |
| bert            | [bert-base-cased](https://huggingface.co/bert-base-cased)                                               | [run_bert.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-classification/run_bert.sh)                       | 910B1  | 783                  | 0.912    |
| bert_japanese   | [bert-base-japanese](https://huggingface.co/cl-tohoku/bert-base-japanese)                               | [run_bert_japanese.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-classification/run_bert_japanese.sh)     | 910B1  | 796                  | 0.8188   |
| bertweet        | [bertweet-base](https://huggingface.co/vinai/bertweet-base)                                             | [run_bertweet.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-classification/run_bertweet.sh)               | 910B1  | 735                  | 0.9346   |
| big_bird        | [bigbird-roberta-base](https://huggingface.co/google/bigbird-roberta-base)                              | [run_big_bird.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-classification/run_big_bird.sh)               | 910B1  | 724                  | 0.9438   |
| bigbird_pegasus | [google/bigbird-pegasus-large-bigpatent](https://huggingface.co/google/bigbird-pegasus-large-bigpatent) | [run_bigbird_pegasus.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-classification/run_bigbird_pegasus.sh) | 910B1  | 331                  | 0.8991   |
| bort            | [amazon/bort](https://huggingface.co/amazon/bort)                                                       | [run_bort.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-classification/run_bort.sh)                       | 910B1  | 1343                 | 0.7729   |
| camembert       | [camembert-base](https://huggingface.co/camembert)                                                      | [run_camembert.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-classification/run_camembert.sh)             | 910B1  | 774                  | 0.8544   |
| canine          | [canine-s](https://huggingface.co/google/canine-s)                                                      | [run_canine.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-classification/run_canine.sh)                   | 910B1  | 540                  | 0.8222   |
| convbert        | [convbert-base-turkish-mc4-cased](https://huggingface.co/dbmdz/convbert-base-turkish-mc4-cased)         | [run_convbert.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-classification/run_convbert.sh)               | 910B1  | 418                  | 0.8452   |
| distilbert      | [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)                               | [run_distilbert.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-classification/run_distilbert.sh)           | 910B1  | 843                  | 0.8968   |

## 版本说明
### 变更
- 2023.06.26：Transformers版本更新到v4.30.2
- 2023.03.05: 首次发布

### FAQ
- 执行`run_bert_japanses.sh`需要安装依赖 `ipadic` 和 `fugashi`

