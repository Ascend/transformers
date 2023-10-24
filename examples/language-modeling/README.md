# Language modeling

本项目展示如何在Ascend NPU下运行Tansformers的[language-modeling](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling)任务。

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

  3、安装language-modeling任务所需依赖
  ```text
  pip3 install -r requirements.txt
  ```

### 准备数据集
language-modeling示例使用的数据集由Hugging Face提供的接口自动下载无需额外准备

### 准备预训练权重
预训练权重类路径:https://huggingface.co/models
请按需下载相应的权重并置于`run_clm.py`同级目录下。

### 准备Metrics所需的脚本
下载地址：
```text
  https://github.com/huggingface/evaluate/tree/main/metrics
```
下载accuracy文件夹放在run_clm.py相同目录下

## 开始训练
执行
```bash
bash ./scripts/run_XXX.sh
```

## 训练结果

| Architecture | Pretrained Model                    | Script                                                                                                           | Device| Performance(8-cards) | Accuracy |
|--------------|-------------------------------------|------------------------------------------------------------------------------------------------------------------|-------|----------------------|----------|
| gpt2         | [gpt2](https://huggingface.co/gpt2) | [run_bert.sh](https://gitee.com/ascend/transformers/tree/develop/examples/language-modeling/scripts/run_gpt2.sh) | NPU | 96                   | 0.3987   |



### FAQ
- 权重例如gpt2文件夹要放在与执行脚本run_clm.py相同的路径下

