# Language model training

Fine-tuning (or training from scratch) the library models for language modeling on a text dataset for GPT, GPT-2,
ALBERT, BERT, DistilBERT, RoBERTa, XLNet... GPT and GPT-2 are trained or fine-tuned using a causal language modeling
(CLM) loss while ALBERT, BERT, DistilBERT and RoBERTa are trained or fine-tuned using a masked language modeling (MLM)
loss. XLNet uses permutation language modeling (PLM), you can find more information about the differences between those
objectives in our [model summary](https://huggingface.co/transformers/model_summary.html).

The following examples, will run on datasets hosted on our [hub](https://huggingface.co/datasets) or with your own
text files for training and validation. We give examples of both below.

## GPT-2/GPT and causal language modeling

The following example fine-tunes GPT-2 on WikiText-2. We're using the raw WikiText-2 (no tokens were replaced before
the tokenization). The loss here is that of causal language modeling.
### Single-card Training

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --use_ascend \
    --use_combine_grad \
    --use_npu16 true \
    --npu_fp16_opt_level O2 \
    --output_dir /tmp/test-clm
```

This takes about xx minutes to train on a single NPU. It reaches
a score of ~20 perplexity once fine-tuned on the dataset.

To run on your own training and validation files, use the following command:

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --use_ascend \
    --use_combine_grad \
    --use_npu16 true \
    --npu_fp16_opt_level O2 \
    --output_dir /tmp/test-clm
```
### Multi-card Training
```bash
python -m torch.distributed.launch --nproc_per_node 8 run_clm.py \
    --model_name_or_path gpt2 \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --use_ascend \
    --use_combine_grad \
    --use_npu16 true \
    --npu_fp16_opt_level O2 \
    --output_dir /tmp/test-clm
```
This takes about xx minutes to train on 8 NPUs. It reaches a perplexity of xx once fine-tuned on the dataset.

## RoBERTa/BERT/DistilBERT and masked language modeling

The following example fine-tunes RoBERTa on WikiText-2. Here too, we're using the raw WikiText-2. The loss is different
as BERT/RoBERTa have a bidirectional mechanism; we're therefore using the same loss that was used during their
pre-training: masked language modeling.

In accordance to the RoBERTa paper, we use dynamic masking rather than static masking. The model may, therefore,
converge slightly slower (over-fitting takes more epochs).
### Single-card Training

```bash
python run_mlm.py \
    --model_name_or_path roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --use_ascend \
    --use_combine_grad \
    --use_npu16 true \
    --npu_fp16_opt_level O2 \
    --output_dir /tmp/test-mlm
```

To run on your own training and validation files, use the following command:

```bash
python run_mlm.py \
    --model_name_or_path roberta-base \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --use_ascend \
    --use_combine_grad \
    --use_npu16 true \
    --npu_fp16_opt_level O2 \
    --output_dir /tmp/test-mlm
```

If your dataset is organized with one sample per line, you can use the `--line_by_line` flag (otherwise the script
concatenates all texts and then splits them in blocks of the same length).

**Note:** On NPU, you should use the flag `--pad_to_max_length` in conjunction with the `--line_by_line` flag to make
sure all your batches have the same length.
### Multi-card Training
```bash
python -m torch.distributed.launch --nproc_per_node 8 run_mlm.py \
    --model_name_or_path roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --use_ascend \
    --use_combine_grad \
    --use_npu16 true \
    --npu_fp16_opt_level O2 \
    --output_dir /tmp/test-mlm
```

## Creating a model on the fly

When training a model from scratch, configuration values may be overridden with the help of `--config_overrides`:


```bash
python run_clm.py \ 
    --model_type gpt2 
    --tokenizer_name gpt2 \ 
    --config_overrides="n_embd=1024,n_head=16,n_layer=48,n_positions=102" \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --use_ascend \
    --use_combine_grad \
    --use_npu16 true \
    --npu_fp16_opt_level O2 \

```
