# Multiple-choice

本项目展示如何在Ascend NPU下运行Tansformers的[multiple-choice](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice)任务。

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

  3、安装multiple-choice任务所需依赖
  ```text
  pip3 install -r requirements.txt
  ```

### 准备数据集
multiple-choice示例使用的数据集由Hugging Face提供的接口自动下载无需额外准备

### 准备预训练权重
预训练权重类路径:https://huggingface.co/models
请按需下载相应的权重并置于`run_swag.py`同级目录下。

## 开始训练
执行
```bash
bash ./scripts/run_XXX.sh
```

## 训练结果

| Architecture    | Pretrained Model                                                                                        | Script                                                                                                         | Device | Performance(8-cards) | Accuracy |
|-----------------|---------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|--------|----------------------|----------|
| bert            | [bert-base-cased](https://huggingface.co/bert-base-cased)                                               | [run_bert.sh](https://gitee.com/ascend/transformers/tree/develop/examples/multiple-choice/scripts/run_bert.sh) | 910B1  | 453                  | 0.7987   |



### FAQ
- 权重例如bert-base-cased文件夹要放在与执行脚本run_swag.py相同的路径下

