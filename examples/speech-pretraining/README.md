# Speech-pretraining

本项目展示如何在Ascend NPU下运行Tansformers的[speech-pretraining](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-pretraining)任务。

## 准备训练环境
### 准备环境
- 环境准备指导。
  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  当前基于 PyTorch 2.0.1 完成测试。
- 安装依赖
  
  1、使用 NPU 设备源码安装Huggingface的Github仓的 accelerate
  ```text
  git clone https://github.com/huggingface/accelerate
  cd accelerate
  pip3 install -e .
  ```
  2、使用 NPU 设备源码安装 Transformers 插件
  ```text
  git clone https://github.com/huggingface/transformers
  cd transformers
  pip3 install -e .
  ```

  3、安装speech-pretraining任务所需依赖
  ```text
  pip3 install -r requirements.txt
  ```

### 准备数据集
speech-pretraining示例使用的数据集由Hugging Face提供的接口自动下载无需额外准备

### 准备预训练权重
预训练权重类路径:https://huggingface.co/models
请按需下载相应的权重并置于`run_wav2vec2_pretraining_no_trainer.py`同级目录下。

## 开始训练
执行
```bash
bash ./scripts/run_XXX.sh
```

## 训练结果

| Architecture | Pretrained Model                                                                                          | Script                                                                                                                            | Device    | Val_loss  |
|--------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|-----------|-----------|
| wav2vec2     | [wav2vec2-base-v2](https://huggingface.co/patrickvonplaten/wav2vec2-base-v2)                              | [run_wav2vec2.sh](https://gitee.com/ascend/transformers/tree/develop/examples/speech-pretraining/scripts/run_wav2vec2.sh)         | 910B1     | 4.366e+00 |



### FAQ
- 权重例如wav2vec2-base-v2文件夹要放在与执行脚本run_wav2vec2_pretraining_no_trainer.py相同的路径下

