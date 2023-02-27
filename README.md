# Optimum Ascend
🤗 Optimum Ascend is the interface between the 🤗 Transformers and and Ascend's nerual-network processing (NPU). It provides a set of tools enabling easy model loading, training and inference on single- and multi-NPU settings for different downstream tasks. The list of officially validated models and tasks is available [here](). Users can try other models and tasks with only few changes.

## What is a Ascend Nerual-Network Processing Unit (NPU)?
> Huawei Ascend NPU is tailored for deep neural network computing. Thanks to the high-performance 3D cube matrix computing units, the Ascend NPU can achieve significant computing power and efficiency. Each matrix computing unit can handle 4096 multiplication and addition calculations within a single instruction cycle.

## Install
To install the latest release of this package:
```text
pip install optimum-ascend
```
Optimum Ascend is fast-moving project, and you may want to install it from source:
```text
pip install git+https://github.com/huggingface/optimum-ascend.git
```

### Installing in developer mode
If you are working on the `optimum-ascend` code then you should use an editable install.
```text
git clone https://github.com/huggingface/optimum-ascend
cd optimum-ascend
pip install -e .
```

## Running the examples
There are a number of examples provided in the `example` directory. Each of these contains a README with command lines for running then on NPU with Optimum Ascend. Don't for get to install the requirements for every example:
```text
cd <example-folder>
pip install -r requirements.txt
```

## How to use it?
🤗 Optimum Ascend was designed with one goal in mind: **to make training and evaluation straightforward for any 🤗 Transformers user while leveraging the complete power of Ascend NPU**.

There are two main classes one needs to know:
- [NPUTrainer](): the trainer class that takes care of distributing the model to run on NPUs, and performing training and evaluation.
- [NPUTrainingArguments](): the class that enables to configure Ascend Mixed Precision and to decide whether optimized operators and optimizers should be used or not.

The [NPUTrainer]() is very similar to the 🤗 [Transformers Trainer](), and adapting a script using the Trainer to make it work with NPU will mostly consist in simply swapping the `Trainer` class for the `NPUTrainer` one. That's how most of the [example scripts]() were adapted from their [original counterparts]().

Original script:
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
  # training arguments...
)

# A lot of code here

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,  # Original training arguments.
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
```
Transformed version that can run on NPU:
```diff
- from transformers import Trainer, TrainingArguments
+ from optimum.ascend import NPUTrainer, NPUTrainingArguments

- training_args = TrainingArguments(
+ training_args = NPUTrainingArguments(
  # training arguments...
+ use_ascend=True,
+ npu_fp16=True, # whether to use mixed precision training
)

# A lot of code here

# Initialize our Trainer
- trainer = Trainer(
+ trainer = NPUTrainer(
    model=model,
    args=training_args,  # Original training arguments.
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
```

Another easy way is to use the `transfor_to_npu` script to auto-replace the original API:
```diff
+ from optimum.ascend import transfor_to_npu
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
  # training arguments...
)

# A lot of code here

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,  # Original training arguments.
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
```

for more information, check our [documentation]().

## Supported Models
The following model architectures and tasks are supported by 🤗 Optimum Ascend.

## NPU Setup
please refer to Ascend NPU's offical [installation guild]().