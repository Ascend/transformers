# Text Classification Examples
## GLUE tasks
Based on the script `run_glue.py`.

Fine-tuning the library models for sequence classification on the GLUE benchmark: [General Language Understanding Evaluation](https://gluebenchmark.com). This script can fine-tune any of the models on the [hub](https://huggingface.co/models) and can also be used for a dataset hosted on our [hub](https://huggingface.co/datasets) or your own data in a csv or a JSON file (the script might need some tweaks in that case, refer to the comments inside for help).

GLUE is made up of a total of 9 different tasks where the task name can be cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte or wnli.

## Fine-tuning BERT on SST2

### Single-card Training
The following example fine-tunes BERT Large on the sst2 dataset hosted on the [hub](https://huggingface.co/datasets):

```bash
python run_glue.py \
  --model_name_or_path bert-large-cased \
  --task_name sst2 \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 128 \
  --output_dir ./output/sst2/ \
  --use_ascend \
  --use_combine_grad \
  --npu_fp16 \
  --npu_fp16_opt_level O2 \
```

> If your model classification head dimensions do not fit the number of labels in the dataset, you can specify `--ignore_mismatched_sizes` to adapt it.

### Multi-card Training
Here is how you would fine-tune the BERT large model (with whole word masking) on the text classification SST2 task using the `run_glue` script, with 8 NPUs:
```bash
python run_glue.py \
  --model_name_or_path bert-large-cased \
  --task_name sst2 \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 128 \
  --output_dir ./output/sst2/ \
  --use_ascend \
  --use_combine_grad \
  --npu_fp16 \
  --npu_fp16_opt_level O2 \
```

> If your model classification head dimensions do not fit the number of labels in the dataset, you can specify `--ignore_mismatched_sizes` to adapt it.

