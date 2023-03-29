# Question answering

本项目展示如何在诸如SQuAD的问答数据集上微调Transformers模型。


## 在SQuAD1.0上微调BERT模型

脚本 [`run_qa.py`](https://gitee.com/ascend/transformers/blob/optimum/examples/question-answering/run_qa.py) 可以微调Hugging Face官方[hub](https://huggingface.co/models)上任意包含`ForQuestionAnswering`结构的模型。

**注意：**
- 该脚本只能用于微调包含fast tokenizer的模型。
- 如果你的数据集包含无答案的样本例如SQuADv2，使用该脚本时请传入参数`--version_2_with_negative`。

下面展示在SQuAD1.0上微调BERT：

```bash
python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --fp16 \
  --fp16_opt_level O2 \
  --use_combine_grad \
  --output_dir /tmp/debug_squad/
```

## 训练结果
**[albert-base-v2](https://huggingface.co/albert-base-v2)在squad上训练结果展示表**

| albert-base-v2  | F1     | FPS     | AMP_Type | Epochs |
|-----------------|--------|---------|----------|--------|
| 1p-V竞品          | 90.59% | 91.084  | O2       | 3      |
| 8p-V竞品          | 90.43% | 993.087 | O2       | 3      |
| 1p-NPU(910ProB) | 91.09% | 83.509  | O2       | 3      |
| 8p-NPU(910Prob) | 90.37% | 803.981 | O2       | 3      |


## 版本说明
### 变更
- 2023.03.05: 首次发布
