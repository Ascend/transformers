# Transformers
## 简介
本项目用于承载Transformers测试用例和模型训练脚本。
当前Transformers已经原生支持使用Ascend NPU训练微调模型（[见](https://github.com/huggingface/transformers/pull/24879)）。

## 支持的特性
当前支持特性：
- 单卡训练
- 多卡训练
- fp16(native amp/apex)

不支持：
- deepspeed
- PyTorch FSDP
- Torch dynamo

## 版本说明
### 依赖
- [accelerate](https://github.com/huggingface/accelerate)

### 变更
[23/08/22] 增加tests框架
[23/07/19] 初次发布

## FQA