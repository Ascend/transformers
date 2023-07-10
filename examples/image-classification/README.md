# Image classification

本项目展示如何在Ascend NPU下运行Tansformers的[image-classification](https://github.com/huggingface/transformers/tree/v4.30.2/examples/pytorch/image-classification)任务。

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
image-classification示例使用的数据集由Hugging Face提供的接口自动下载无需额外准备，如果出现数据集和metric预处理脚本下载失败的问题，请手动下载
- metric：把[accuracy](https://github.com/huggingface/evaluate/tree/main/metrics/accuracy)放到`image-classification`目录下
- datasets: 把[beans](https://huggingface.co/datasets/beans)放到`image-classificatoin`目录下
```text
 image-classification
    ├── accuracy
    │   ├── accuracy.py
    │   └── app.py
    ├── beans
    │   ├── data
    │   │    ├── test.zip
    │   │    ├── train.zip 
    │   │    └── validation.py
    │   ├── dataset_infos.json
    │   └── beans.py
    └── ... 
```

### 准备预训练权重
官方[audio-classification](https://github.com/huggingface/transformers/tree/v4.30.2/examples/pytorch/audio-classification)中`run_image_classificaton.py`和`run_image_classificaton_no_trainer.py`使用的预训练权重为[vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)，请按需下载相应的权重并置于`run_image_classification.py`同级目录下。

## 开始训练
- 使用Trainer类的用例
  ```text
  bash ./test/run_image_classification.sh
  ```
- 不使用Trainer类通过accelerate启动的用例
  ```text
  bash ./test/run_image_classification_no_trainer.sh
  ```

## 训练结果

| Architecture | Pretrained Model                                                                       | Script                                                                                                                                 | supported  | 
|--------------|----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|------------|
| vit          | [vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) | [run_image_classification.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-generation/run_image_classification.sh) | ✔️         |




## 版本说明
### 变更
- 2023.06.26：Transformers版本更新到v4.30.2
- 2023.03.05: 首次发布

