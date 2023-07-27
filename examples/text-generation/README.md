# Text generation

本项目展示如何在Ascend NPU下运行Tansformers的[text-generation](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-generation)任务。

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

  3、安装text-generation任务所需依赖
  ```text
  pip3 install -r requirements.txt
  ```

### 准备数据集
无需准备

### 准备预训练权重
预训练权重类路径:https://huggingface.co/models
请按需下载相应的权重并置于`run_generation.py`同级目录下。

## 开始训练
执行
```bash
bash ./scripts/run_XXX.sh
```



### FAQ


