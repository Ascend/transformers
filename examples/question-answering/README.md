# Question answering

本项目展示如何在Ascend NPU下运行Tansformers的[question_answering](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)任务。

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

  3、安装question-answering任务所需依赖
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
请按需下载相应的权重并置于`run_qa.py`同级目录下。

### 准备Metrics所需的脚本
下载地址：
```text
  https://github.com/huggingface/evaluate/tree/main/metrics
```
在run_qa.py相同目录下新建文件夹metrics ，下载squad、squad_v2文件夹放在metrics文件夹中

## 开始训练
执行
```bash
bash ./scripts/run_XXX.sh
```

## 训练结果

| Architecture | Pretrained Model                                          | Script                                                                                                            | Device | Performance(8-cards) | Exact_match | F1       | Performance(8-cards)(竞品A) |  Exact_match(竞品A)  | F1（竞品A）| 
|--------------|-----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|--------|----------------------|-------------|----------|---------------------------| -------------------|--------|
| bert         | [bert-base-cased](https://huggingface.co/bert-base-cased) | [run_bert.sh](https://gitee.com/ascend/transformers/tree/develop/examples/question-answering/scripts/run_bert.sh) | NPU  | 429                  | 0.7957      | 0.8660   | 306                       |  0.7961            | 0.8736 |



### FAQ
- 权重例如bert-base-cased文件夹要放在与执行脚本run_qa.py相同的路径下
- 需要将run_qa.py中601行代码进行修改：
  ```
  metric = evaluate.load("squad_v2" if data_args.version_2_with_negative else "squad")
  ```
  修改为：
  ```
  metric = evaluate.load("./metrics/squad_v2" if data_args.version_2_with_negative else "./metrics/squad")
  ```
