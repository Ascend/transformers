# coding=utf-8
# Copyright 2023-present Huawei Technologies Co., Ltd. All rights reserved.
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Testing suite for the Prefix_Tuning of Peft. """

from typing import List, Optional
from tqdm import tqdm
from dataclasses import dataclass, field

import torch
try:
    import torch_npu  # 必填，若不加torch_npu则会出现精度严重劣化问题
except ImportError:
    print("This is not a device of NPU.")
from torch.optim import AdamW
from torch.utils.data import DataLoader

import evaluate
from accelerate import PartialState
from datasets import load_dataset
from peft import (
    PrefixTuningConfig,
    get_peft_model,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    get_linear_schedule_with_warmup,
)


@dataclass
class TrainingArguments:
    batch_size: Optional[int] = field(default=32, metadata={"help": "Batch size for training."})
    num_epochs: Optional[int] = field(default=3, metadata={"help": "Epochs size for training."})
    learning_rate: Optional[float] = field(default=1e-2, metadata={"help": "Learning rate for training."})


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="bert-base-cased", metadata={"help": "Checkpoint for training."})
    task: Optional[str] = field(default="mrpc", metadata={"help": "Dataset task for training."})
    datasets_name: Optional[str] = field(default="glue", metadata={"help": "Datasets name for training."})
    metric_name: Optional[str] = field(default="metrics/glue", metadata={"help": "metric name for training."})
    remove_columns: Optional[List[str]] = field(
        default_factory=lambda: ["idx", "sentence1", "sentence2"], metadata={"help": "remove columns of datasets."}
    )
    original_column_name: Optional[str] = field(
        default="label", metadata={"help": "original column name to be replaced."}
    )
    new_column_name: Optional[str] = field(default="labels", metadata={"help": "new column name to replace."})


@dataclass
class PeftConfigsArguments:
    task_type: Optional[str] = field(default="SEQ_CLS", metadata={"help": "task type for peft config."})
    num_virtual_tokens: Optional[int] = field(default=20, metadata={"help": "The number of virtual tokens to use."})


def main():
    parser = HfArgumentParser((TrainingArguments, ModelArguments, PeftConfigsArguments))
    training_args, model_args, peft_config_args = parser.parse_args_into_dataclasses()

    peft_config = PrefixTuningConfig(
        task_type=peft_config_args.task_type,
        num_virtual_tokens=peft_config_args.num_virtual_tokens
    )

    if any(k in model_args.model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets = load_dataset(model_args.datasets_name, model_args.task)
    metric = evaluate.load(model_args.metric_name, model_args.task)

    def tokenize_function(examples):
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=model_args.remove_columns,
    )

    tokenized_datasets = tokenized_datasets.rename_column(model_args.original_column_name, model_args.new_column_name)

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=training_args.batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=training_args.batch_size
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, return_dict=True)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    optimizer = AdamW(params=model.parameters(), lr=training_args.learning_rate)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(train_dataloader) * training_args.num_epochs),
        num_training_steps=(len(train_dataloader) * training_args.num_epochs),
    )

    model.to(PartialState().default_device)
    for epoch in range(training_args.num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch.to(PartialState().default_device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch.to(PartialState().default_device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        print(f"epoch {epoch}:", eval_metric)


if __name__ == "__main__":
    main()
