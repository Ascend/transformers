# Text classification examples

## GLUE tasks

[`run_glue.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py)脚本用于训练sequence classification任务，GLUE由9个不同的子任务组成。下面是在其中一个子任务上运行的脚本：

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