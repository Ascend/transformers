# Contrastive-Image-Text

本项目展示如何在Ascend NPU下运行Tansformers的[icontrastive-image-text](https://github.com/huggingface/transformers/tree/main/examples/pytorch/contrastive-image-text) 任务。

## 准备训练环境

### 准备环境

- 环境准备指导。 请参考《[Pytorch框架训练环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fptes) 》。 当前基于 PyTorch 2.0.1 完成测试。

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

  3、安装contrastive-image-text任务所需依赖

  ```
  pip3 install -r requirements.txt
  ```

### 准备数据集
Contrastive-Image-Text示例使用的数据集是coco数据集
* 下载coco数据压缩包
```bash
mkdir coco
cd coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
cd ..
```
* 下载coco数据脚本 [ydshieh/coco_dataset_script](https://huggingface.co/datasets/ydshieh/coco_dataset_script) 放在run_clip.py 同目录下

* 执行下列python代码，下载并准备数据集

```python
import os
import datasets

COCO_DIR = os.path.join(os.getcwd(), "coco")
ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=COCO_DIR)
```

### 准备预训练权重

创建[VisionTextDualEncoderModel](https://huggingface.co/docs/transformers/model_doc/vision-text-dual-encoder#visiontextdualencoder)
```python
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoTokenizer,
    AutoImageProcessor
)

model = VisionTextDualEncoderModel.from_vision_text_pretrained(
    "openai/clip-vit-base-patch32", "roberta-base"
)

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

# save the model and processor
model.save_pretrained("clip-roberta")
processor.save_pretrained("clip-roberta")
```
tips: 如果模型 openai/clip-vit-base-patch32 和 roberta-base 自动下载失败，可在[huggingface dataset](https://huggingface.co/datasets)
中手动下载，并放至run_clip.py 的同一目录下。

## 开始训练

执行下面命令

```bash
bash ./scripts/run_clip.sh
```

## 训练结果

| Architecture |           Pretrained Model            |   Script    | Device  | Performance(8-cards) | Train Loss | Pretraining Times |
|:------------:|:-------------------------------------:|:-----------:|:-------:|:--------------------:|:----------:|:-----------------:|
|     clip     | clip-vit-base-patch32 —— roberta-base | run_clip.sh |  910B1  |        415.05        |   0.3576   |      47mins       |

