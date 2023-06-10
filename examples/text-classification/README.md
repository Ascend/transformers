# Text classification

## GLUE任务

 `run_glue.py` 脚本用于训练sequence classification任务，GLUE由9个不同的任务组成。以下是在其中一个上运行的脚本：

```bash
export TASK_NAME=sst2

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --use_ascend \
  --npu_fp16 \
  --npu_fp16_opt_level O2 \
  --use_combine_grad \
  --output_dir /tmp/$TASK_NAME/
```
其中`task_name`可以是 `cola`、`sst2`、`mrpc`、`stsb`、`qqp`、`mnli`、`qnli`、`rte`、`wnli`。

> 如果您的模型的classification head维度和数据集labels数不匹配，您可以通过指定`--ignore_mismatched_sizes`忽略不匹配项。

## 训练结果
**[bert-base-cased](https://huggingface.co/bert-base-cased)在sst2上训练结果展示表**

| bert-base-cased | Acc    | FPS      | AMP_Type | Epochs |
|-----------------|--------|----------|----------|--------|
| 1p-V竞品          | 91.44% | 303.068  | O2       | 3      |
| 8p-V竞品          | 91.63% | 1483.905 | O2       | 3      |
| 1p-NPU(910ProB) | 92.48% | 305.27   | O2       | 3      |
| 8p-NPU(910Prob) | 91.4%  | 1672.478 | O2       | 3      |

**[bert-large-uncased](https://huggingface.co/bert-base-uncased)在sst2上训练结果展示表**

| bert-large-uncased | Acc    | FPS     | AMP_Type | Epochs |
|--------------------|--------|---------|----------|--------|
| 1p-V竞品             | 93.17% | 103.237 | O2       | 3      |
| 8p-V竞品             | 92.43% | -       | O2       | 3      |
| 1p-NPU(910ProB)    | 93.29% | 117.2   | O2       | 3      |
| 8p-NPU(910Prob)    | 93.12% | 694.745 | O2       | 3      |

## 版本说明
### 变更
- 2023.03.05: 首次发布
