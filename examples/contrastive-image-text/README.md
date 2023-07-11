# VisionTextDualEncoder and CLIP model training examples

本项目展示如何在Ascend NPU下运行Tansformers的[contrastive-image-text](https://github.com/huggingface/transformers/tree/v4.30.2/examples/pytorch/contrastive-image-text)任务。

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

  3、安装contrastive-image-text任务所需依赖
  ```text
  pip3 install -r requirements.txt
  ```

### 准备数据集
下载 COCO dataset(2017)
```bash
mkdir data
cd data
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
cd ..
```
使用[coc_dataset_script](https://huggingface.co/datasets/ydshieh/coco_dataset_script)处理下载的数据集：
```python
import os
import datasets

COCO_DIR = os.path.join(os.getcwd(), "data")
ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=COCO_DIR)
```

### 使用vision encoder 模型和text encoder模型生成所需模型
生成 [VisionTextDualEncoderModel](https://huggingface.co/docs/transformers/model_doc/vision-text-dual-encoder#visiontextdualencoder).

这里给出如何加载模型权重（使用预训练的视觉和文本模型）的示例：

```python3
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

## 开始训练
执行 `run_clip.sh`

## 训练结果
| Architecture | Pretrained Model    | Script                                                                                                        | supported | 
|--------------|---------------------|---------------------------------------------------------------------------------------------------------------|-----------|
| clip         | manually-generation | [run_clip.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/contrastive-image-text/run_clip.sh) | ✔️        |


## 版本说明
### 变更
- 2023.06.26：Transformers版本更新到v4.30.2
- 2023.03.05: 首次发布

### FAQ
- `RuntimeError: No such operator image::read_file`
  > 请通过源码编译安装 torchvision
