# PEFT

## 安装PEFT所需的相关依赖
```
cd transformers/tests/peft/
pip install -r requirements.txt
```

## 准备预训练模型

- 下载预训练模型[bert-base-cased](https://huggingface.co/bert-base-cased), 路径为transformers/tests/peft/bert-base-cased
- 下载数据集文件[glue](https://huggingface.co/datasets/glue), 路径为transformers/tests/peft/glue
- 下载metrics文件[glue](https://github.com/huggingface/evaluate/tree/main/metrics/glue), 路径为transformers/tests/peft/metrics/glue

## 执行脚本
```
python3 test_XXX.py
```