# Image classification

本项目展示如何在Ascend NPU下运行Tansformers的[image classification](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)任务。



## 准备训练环境

### 准备环境

- 环境准备指导。 请参考《[Pytorch框架训练环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fptes)》。 当前基于 PyTorch 2.0.1 完成测试。

- 安装依赖

  1、使用 NPU 设备源码安装Huggingface的Github仓的 accelerate

  ```
  git clone https://github.com/huggingface/accelerate
  cd accelerate
  pip3 install -e .
  ```

  2、使用 NPU 设备源码安装 Transformers 插件

  ```
  git clone https://github.com/huggingface/transformers
  cd transformers
  pip3 install -e .
  ```

  3、安装image-classification任务所需依赖

  ```
  pip3 install -r requirements.txt
  ```

### 准备数据集

image-classification示例使用的数据集由Hugging Face提供的接口自动下载无需额外准备

### 准备预训练权重

预训练权重类路径:[huggingface-hub](https://huggingface.co/models) 请按需下载相应的权重并在`scripts/run_image_classification.sh`中设置`--model_name_or_path`

## 开始训练

执行

```
bash ./scripts/run_image_classification.sh
```

## 训练结果

| Architecture | Pretrained Model           | Script                      | Device | Performance(8-cards) | Accuracy |
| ------------ | -------------------------- | --------------------------- | ------ | -------------------- | -------- |
| ViT          | vit-base-patch16-224-in21k | run_image_classification.sh | 910B1  | 69.469               | 0.985    |

### FAQ

- `--model_name_or_path`是权重文件夹的路径