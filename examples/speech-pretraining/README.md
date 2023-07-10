# Speech Recognition Pre-Training

本项目展示如何在Ascend NPU下运行Tansformers的[speech-pretraining](https://github.com/huggingface/transformers/tree/v4.30.2/examples/pytorch/speech-pretraining)任务。

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

  3、安装speech-pretraining任务所需依赖
  ```text
  pip3 install -r requirements.txt
  ```

### 准备数据集&metric
speech-pretraining示例使用的数据集由Hugging Face提供的接口自动下载无需额外准备，如果出现数据集和metric预处理脚本下载失败的问题，请手动下载

### 准备预训练权重


## 开始训练
```text
bash ./test/run_xxx.sh
```

## 训练结果
| Architecture       | Pretrained Model | Script                                                                                                                                            | supported | 
|--------------------|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| wav2vec2           | pretraining      | [run_wav2vec2_pretraining](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/speech-pretraining/run_wav2vec2_pretraining_no_trainer.sh) | ✔️        |

## 版本说明
### 变更
- 2023.06.26：Transformers版本更新到v4.30.2
- 2023.03.05: 首次发布
