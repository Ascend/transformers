# Translation

本项目展示如何在翻译任务上微调和评估transformers模型。

## 微调T5

下面展示如何在wmt16上微调T5-small:

```bash
python examples/pytorch/translation/run_translation.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --source_prefix "translate English to Romanian: " \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --fp16 \
    --fp16_opt_level O2 \
    --use_combine_grad \
    --predict_with_generate
```
更多用法请参考官方[示例](https://github.com/huggingface/transformers/examples/pytorch/translation)。

## 训练结果
**[t5-small](https://huggingface.co/t5-small)在wmt16上训练结果展示表**

| t5-small        | BLEU    | FPS     | AMP_Type | Epochs |
|-----------------|---------|---------|----------|--------|
| 1p-V竞品          | 26.4444 | 61.443  | O2       | 3      |
| 8p-V竞品          | 26.3522 | 288.524 | O2       | 3      |
| 1p-NPU(910ProB) | 26.3213 | 33.003  | O2       | 3      |
| 8p-NPU(910Prob) | 26.2331 | 214.161 | O2       | 3      |


## 版本说明
### 变更
- 2023.03.05: 首次发布
- 2023.04.04: eval阶段走二进制解决算子编译超时导致的掉卡问题
