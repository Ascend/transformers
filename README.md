# transformers_npu

本项目开发了 transformers 插件，用于晟腾适配 transformers 套件。

## 版本说明

目前仅支持 transformers 4.18.0 版本：https://github.com/huggingface/transformers/tree/v4.18.0

不支持通过huggingface/accelerate加速模型训练。

## 安装方法

- 当前套件支持的固件与驱动、CANN 以及 PyTorch 如下表所示

  **表 1** 版本配套表

  | 配套 | 版本  |
  |-----| ---- |
  |固件与驱动| [22.0.RC3](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN | [6.1.RC1](https://www.hiascend.com/software/cann/commercial?version=6.1.RC1) |
  | PyTorch | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/) |

- 环境准备

  请参考《[PyTorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》

- 安装原生 transformers 套件

  ```python
  pip3 install transformers==4.18.0
  ```
  
- 下载并安装 transformers_npu 插件

  ```python
  git clone -b 4.18.0 https://gitee.com/ji-huazhong/transformers_npu
  cd transformers_npu
  pip3 install ./
  ```

## 插件使用方法
在引入 `import transformers` 之前 `import transformers_npu`，参考 `examples/question-answering/run_qa.py`
```python
...
import transformers_npu
import transformers
...
```

## 运行下游任务
请参考 `examples` 的下游任务用例，例如[问答任务](examples/question-answering/README.md)。