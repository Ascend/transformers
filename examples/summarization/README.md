# Summarization

本项目展示如何在Ascend NPU下运行Tansformers的[summarization](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization)任务。

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

  3、安装summarization任务所需依赖
  ```text
  pip3 install -r requirements.txt
  ```

### 准备数据集
数据集下载地址：
```text
  https://huggingface.co/datasets
```

### 准备预训练权重
预训练权重类路径:https://huggingface.co/models
请按需下载相应的权重并置于`run_summarization.py`同级目录下。

### 准备Metrics所需的脚本
下载地址：
```text
  https://github.com/huggingface/evaluate/tree/main/metrics
```
下载rouge文件夹放在run_summarization.py相同目录下

## 开始训练
执行
```bash
bash ./scripts/run_XXX.sh
```

## 训练结果

| Architecture | Pretrained Model                                    | Script                                                                                                    | Device | Performance(8-cards) | Rouge1 | Rouge2 | RougeL | RougeLsum |
|--------------|-----------------------------------------------------|-----------------------------------------------------------------------------------------------------------|--------|----------------------|--------|--------|--------|-----------|
| t5           | [t5-small](https://huggingface.co/t5-small)         | [run_t5.sh](https://gitee.com/ascend/transformers/tree/develop/examples/summarization/scripts/run_t5.sh)  | 910B1  | 538                  | 0.4131 | 0.1891 | 0.2924 | 0.3849    |



### FAQ
- 权重例如t5-small文件夹要放在与执行脚本run_summarization.py相同的路径下


