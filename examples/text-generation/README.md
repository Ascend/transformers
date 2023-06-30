## 文本生成

[`run_generation.py`](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-generation/run_generation.py)脚本使用自回归模型: GPT, GPT-2, GPTJ, Trasnformer-XL, XLNet, CTRL, BLOOM, LLAMA, OPT生成文本。
下面给出使用示例：
```bash
python run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=gpt2
```

## 测试脚本

| Architecture | Pretrained Model                                                    | Script                                                                                                             | eval        | 
|--------------|---------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|-------------|
| gpt2         | [gpt2](https://huggingface.co/gpt2)                                 | [run_gpt2.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-generation/run_gpt2.sh)             | ✔️(修改源码后跑通) |
| ctrl         | [ctrl](https://huggingface.co/ctrl)                                 | [run_ctrl.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-generation/run_ctrl.sh)             | ✔️          |
| openai-gpt   | [openai-gpt](https://huggingface.co/openai-gpt)                     | [run_openai_gpt.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-generation/run_openai_gpt.sh) | ✔️          |
| xlnet        | [xlnet-base-cased](https://huggingface.co/xlnet-base-cased)         | [run_xlnet.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-generation/run_xlnet.sh)           | ✔️(修改源码后跑通) |
| transfo-xl   | [transfo-xl-wt103](https://huggingface.co/transfo-xl-wt103)         | [run_transfo_xl.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-generation/run_transfo_xl.sh) | ✖️          |
| xlm          | [xlm-mlm-en-2048](https://huggingface.co/xlm-mlm-en-2048)           | [run_xlm.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-generation/run_xlm.sh)               | ✔️          |
| gptj         | [gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b)              | [run_gptj.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-generation/run_gptj.sh)             ||
| bloom        | [bloom-560m](https://huggingface.co/bigscience/bloom-560m)          | [run_bloom.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-generation/run_bloom.sh)           | ✔️(修改源码后跑通) |
| llama        | [llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf) | [run_llama.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-generation/run_llama.sh)           | ✔️(修改源码后跑通) |
| opt          | [opt](https://huggingface.co/facebook/opt-125m)                     | [run_opt.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/text-generation/run_opt.sh)               | ✔️(修改源码后跑通) |

## 遗留问题
- 二进制下 `transforemrs/generation/utils.py` - line 2676 in samples，算子 `ReduceProd` 报错不支持Bool类型的入参
  > 临时规避：`next_tokens.tile(eos_token_id_tensor.shape[0],1).ne(eos_token_id_tensor.unsequeeze(1)).int().prod(dim=0)`
- bloom模型有如下报错：`modeling_bloom.py` - line 109, in build_alibi_tensor，算子 `Pow`不支持入参为DT_FLOAT和DT_INT32的场景
  > 临时规避：`torch.pow(base.powers.float())` 
   
- transfo-xl npu上存在功能报错待定位，cpu上执行正常