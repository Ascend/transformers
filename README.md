# Optimum Ascend
## 简介
本项目开发了Optimum Ascend插件，用于昇腾NPU适配Transformers套件，为使用Transformers套件的开发者提供昇腾AI处理器的超强算力。当前已经验证的模型和任务可以下在[这里](https://gitee.com/ascend/transformers/tree/optimum#supported-models)查询。用户也可以自行尝试训练其他任务或模型。


## 安装

```text
git clone https://gitee.com/ascend/transformers.git -b optimum && cd transformers
pip install -e .
```
> **说明**：当前配套版本如下：
> - [Transformersv4.25.1](https://github.com/huggingface/transformers/tree/v4.25.1)
> - Torch + torch_npu 1.8.1 + apex

## 使用说明
### NPUTrainer和NPUTrainingArguments类
这里有两个主要的类需要重点说明：
- [NPUTrainer](https://gitee.com/ascend/transformers/blob/optimum/optimum/ascend/trainer.py): trainer类用于在NPU上分布式训练并评估模型。
- [NPUTrainingArguments](https://gitee.com/ascend/transformers/blob/optimum/optimum/ascend/training_args.py): 这个类用于配置是否开启混合精度训练以及具体设置。

[NPUTrainer](https://gitee.com/ascend/transformers/blob/optimum/optimum/ascend/trainer.py) 和原生套件的 🤗 [Transformers Trainer](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py)非常相似，因此大部分使用`Trainer`和`TrainingArguments`的训练脚本只需要用`NPUTrainer`和`NPUTrainingArguments`替换就能工作。

原始脚本：
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
可以在NPU上运行的迁移版本：
```diff
- from transformers import Trainer, TrainingArguments
+ from optimum.ascend import NPUTrainer, NPUTrainingArguments

- training_args = TrainingArguments(
+ training_args = NPUTrainingArguments(
  # training arguments...
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

另一种简单的方式是使用一键迁移脚本`transfor_to_npu`自动替换原始脚本：
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

### 支持的功能
当前支持在昇腾NPU上对Transformers套件进行单机单卡、单机多卡训练以及（由apex插件提供的）混合精度训练。这里以[问答任务](https://gitee.com/ji-huazhong/transformers_test/tree/master/examples/question-answering)为例说明插件的用法。
单卡训练脚本如下：
```bash
python3.7 run_qa.py \
        --model_name_or_path albert-base-v2 \
        --dataset_name squad \
        --do_train \
        --do_eval \
        --num_train_epochs 3 \
        --learning_rate 3e-5 \
        --device_id 0 \
        --half_precision_backend apex \
        --fp16 \
        --fp16_opt_level O2 \
        --optim adamw_apex_fused_npu \
        --use_combine_grad \
        --loss_scale 1024.0 \
        --dataloader_drop_last \
        --output_dir ./output
```
训练脚本参数说明如下：
```text
--output_dir             // 训练结果和checkpoint保存路径
--num_train_epochs       // 训练的epoch次数
--model_name_or_path     // 与训练模型文件夹路径
--dataset_name           // 数据集名称
--do_train               // 执行训练
--do_eval                // 执行评估
--dataloader_drop_last   // 丢弃最后一个不完整的batch
--learning_rate          // 初始学习率
--device_id              // [插件新增参数]指定单卡训练时的卡号，多卡训练无需使用
--half_precision_backend // 指定混合精度使用的后端，当前只支持apex，需要手动指定
--fp16                   // 使用混合精度训练，当前只支持apex
--fp16_opt_level         // apex混合精度级别
--optim                  // 指定使用的优化器，当前可以使用NPU亲和的融合优化器 adamw_apex_fused_npu
--loss_scale             // [插件新增参数]apex混合精度训练使用固定loss_scale，用于性能调优
--use_combine_grad       // [插件新增参数]apex混合精度训练使用combine_grad选项，用于性能调优
```

## 验证的模型
Optimum Ascend插件支持的模型和下游任务见下表。

| Architecture | Single Card | Multi Card | Tasks                                                                 |
|--------------|-------------|------------|-----------------------------------------------------------------------|
| BERT         | ✔️          | ✔️         | <li>[text classification](https://gitee.com/ji-huazhong/transformers_test/blob/master/examples/text-classification/test/train_bert_base_cased_full_8p.sh)</li><br><li>[token classification](https://gitee.com/ji-huazhong/transformers_test/blob/master/examples/token-classification/test/train_bert_base_NER_full_8p.sh)</li> |
| ALBERT       | ✔️          | ✔️         | <li>[question answering](https://gitee.com/ji-huazhong/transformers_test/blob/master/examples/question-answering/test/train_albert_base_v2_full_8p.sh)</li>                                       |                     |
| RoBERTa      | ✔️          | ✔️         | <li>[language modeling](https://gitee.com/ji-huazhong/transformers_test/blob/master/examples/language-modeling/test/train_roberta_base_full_8p.sh)</li>                                        |
| T5           | ✔️          | ✔️         | <li>[translation](https://gitee.com/ji-huazhong/transformers_test/blob/master/examples/translation/test/train_t5_small_full_8p.sh)</li>                                              |
| GPT-2        | ✔️          | ✔️         | <li>[language modeling](https://gitee.com/ji-huazhong/transformers_test/blob/master/examples/language-modeling/test/train_gpt2_full_8p.sh)</li>                                        |
> Transformers套件支持的其他模型或下游任务可能也能被Optimum Ascend插件支持，用户可以自行尝试。

## 配置NPU训练环境
请参考昇腾官方的《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

## 版本说明
### 变更
2023.03.29：首次发布

2023.04.04：translation下游任务eval阶段使用二进制以解决算子编译过长导致的掉卡问题

## FQA