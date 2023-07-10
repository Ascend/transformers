# Image pretraining examples

本项目展示如何在Ascend NPU下运行Tansformers的[image-pretraining](https://github.com/huggingface/transformers/tree/v4.30.2/examples/pytorch/image-pretraining)任务。

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
image-pretraining示例使用的数据集由Hugging Face提供的接口自动下载无需额外准备，如果出现数据集预处理脚本下载失败的问题，请手动下载

datasets: 
- 把[cifar10](https://huggingface.co/datasets/cifar10)放到`image-pretraining`目录下

```text
 image-pretraining
    ├── cifar10
    │   ├── cifar10
    │   └── dataset_infos.json
    └── ... 
```

### 准备工作
本项目从零开始训练无需额外准备预训练模型权重。

如果要预训练 Swin Transformer，参考官方README，先自定义configuration并保存：
```python
from transformers import SwinConfig

IMAGE_SIZE = 192
PATCH_SIZE = 4
EMBED_DIM = 128
DEPTHS = [2, 2, 18, 2]
NUM_HEADS = [4, 8, 16, 32]
WINDOW_SIZE = 6

config = SwinConfig(
    image_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    embed_dim=EMBED_DIM,
    depths=DEPTHS,
    num_heads=NUM_HEADS,
    window_size=WINDOW_SIZE,
)
config.save_pretrained("swin")
```


## 开始训练
执行 `bash test/run_xxx.sh`

## 训练结果
| Architecture | Pretrained Model | Script                                                                                                   | supported  | 
|--------------|------------------|----------------------------------------------------------------------------------------------------------|------------|
| vit          | None             | [run_mim.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/image-pretraining/run_mim.sh)   | ✔️         |
| vit_mae      | None             | [run_mae.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/image-pretraining/run_mae.sh)   | ✔️         |
| swin         | None             | [run_swin.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/image-pretraining/run_swin.sh) | ✔️         |

## 版本说明
### 变更
- 2023.06.26：Transformers版本更新到v4.30.2
- 2023.03.05: 首次发布
