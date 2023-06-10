# Language model training
本任务旨在从零训练或者微调模型，可以在文本数据集上对GPT、GPT-2、ALBERT、BERT、DistilBERT、RoBERTa、XLNet等模型进行语言建模。其中，GPT和GPT-2使用因果语言建模(causal language modeling, CLM)，ALBERT、BERT、DistilBERT和RoBERTa使用掩码语言建模(masked language modeling)，而XLNET使用
permutation language modeling(PLM)。

## GPT-2/GPT and causal language modeling

下面展示如何在WikiText-2上微调GPT-2。我们使用的是原始的WikiText-2数据集(分词前没有词元被替换)：

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --fp16 \
    --fp16_opt_level O2 \
    --use_combine_grad \
    --output_dir ./output
```
在自定义数据集上训练模型：

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --fp16 \
    --fp16_opt_level O2 \
    --use_combine_grad \
    --output_dir ./output
```

### RoBERTa/BERT/DistilBERT and masked language modeling

在原生WikiText-2上微调RoBERTa：

```bash
python run_mlm.py \
    --model_name_or_path roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --fp16 \
    --fp16_opt_level O2 \
    --use_combine_grad \
    --output_dir ./ouput
```
如果你的数据集是按照每行一个样本的形式组织的，你可以使用`--line_by_line`参数（拼接所有文本，然后将其分割成长度相同的块）。

## 训练结果
**[roberta-base](https://huggingface.co/roberta-base)在wikitext上训练结果展示表**

| roberta-base    | Acc    | FPS     | AMP_Type | Epochs |
|-----------------|--------|---------|----------|--------|
| 1p-V竞品          | 0.685  | 227.211 | O2       | 3      |
| 8p-V竞品          | 0.6978 | 989.902 | O2       | 3      |
| 1p-NPU(910ProB) | 0.6917 | 213.038 | O2       | 3      |
| 8p-NPU(910ProB) | 0.6984 | 866.635 | O2       | 3      |

**[gpt2](https://huggingface.co/gpt2)在wikitext上训练结果展示表**

| gpt2            | Acc    | FPS    | AMP_Type | Epochs |
|-----------------|--------|--------|----------|--------|
| 1p-V竞品          | 42.62% | 16.842 | O2       | 3      |
| 8p-V竞品          | 42.06% | -      | O2       | 3      |
| 1p-NPU(910ProB) | 42.62% | 11.392 | O2       | 3      |
| 8p-NPU(910ProB) | 42.25% | 69.126 | O2       | 3      |

## 版本说明
### 变更
- 2023.03.05: 首次发布