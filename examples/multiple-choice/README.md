# Multiple Choice

## Fine-tuning on SWAG with the Trainer

`run_swag` allows you to fine-tune any model from our [hub](https://huggingface.co/models) (as long as its architecture as a `ForMultipleChoice` version in the library) on the SWAG dataset or your own csv/jsonlines files as long as they are structured the same way. To make it works on another dataset, you will need to tweak the `preprocess_function` inside the script.

```bash
python run_swag.py \
--model_name_or_path roberta-base \
--do_train \
--do_eval \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--output_dir /tmp/swag_base \
--per_gpu_eval_batch_size=16 \
--per_device_train_batch_size=16 \
--overwrite_output \
--use_ascend \
--use_combine_grad \
--npu_fp16 \
--npu_fp16_opt_level O2
```
Training with the defined hyper-parameters yields the following results:
```
***** Eval results *****
eval_acc = 0.8253
eval_loss = 0.4593
```
