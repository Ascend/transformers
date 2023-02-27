# SQuAD
Based on the script [`run_qa.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py).

Note: This script only works with models that have a fast tokenizer (backed by the 🤗 Tokenizers library) as it uses special features of those tokenizers. You can check if your favorite model has a fast tokenizer in [this table](https://huggingface.co/transformers/index.html#supported-frameworks).

`run_qa.py` allows you to fine-tune any supported model on the SQUAD dataset or another question-answering dataset of the datasets library or your own csv/jsonlines files as long as they are structured the same way as SQUAD. You might need to tweak the data processing inside the script if your data is structured differently.

Note that if your dataset contains samples with no possible answers (like SQUAD version 2), you need to pass along the flag `--version_2_with_negative`.

## Fine-tuning BERT on SQuAD1.1

### Single-card Training
This example code fine-tunes BERT on the SQuAD1.1 dataset. It runs in xx minutes with BERT-large.
```
python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./output/debug_squad/ \
  --use_ascend \
  --use_npu16 true \
  --npu_fp16_opt_level O2 \
```
Training with the previously defined hyper-parameters yields the following results:
```
f1 =
exact_match =
```

### Multi-card Training
Here is how you would fine-tune the BERT large model (with whole word masking) on the SQuAD dataset using the run_qa script, with 8 NPUs:
```
python -m torch.distributed.launch --nproc_per_node 8 run_qa.py \
    --model_name_or_path bert-large-uncased \
    --dataset_name squad \
    --do_train \
    --do_eval \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir /tmp/squad_output/ \
    --use_ascend \
    --npu_fp16 true \
    --npu_fp16_opt_level O2 \
```
It runs in xx minutes with BERT-large and yields the following results:
```
f1 =
exact_match =
```

