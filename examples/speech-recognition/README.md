# Automatic Speech Recognition Examples

本项目展示如何在Ascend NPU下运行Tansformers的[speech-recognition](https://github.com/huggingface/transformers/tree/v4.30.2/examples/pytorch/speech-recognition)任务。

## 准备训练环境
### 准备环境
- 环境准备指导。
  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  当前基于 PyTorch 1.11 完成测试。
- 安装依赖
  
  1、使用 NPU 设备源码安装适配昇腾的 accelerate
  ```text
  git clone -b accelerate-v0.21.0 https://gitee.com/ascend/transformers.git accelerate
  cd accelerate
  pip3 install -e ./
  ```
  2、使用 NPU 设备源码安装 Transformers 插件
  ```text
  git clone -b v4.30.2 https://gitee.com/ascend/transformers.git
  cd transformers
  pip3 install -e ./
  ```
  > 注：该插件依赖Transformers-v4.30.2，将会自动安装该版本

  3、安装speech-recognition任务所需依赖
  ```text
  pip3 install -r requirements.txt
  ```

### 准备数据集
text-classification示例使用的数据集由Hugging Face提供的接口自动下载无需额外准备

### 准备预训练权重
请从[huggingface hub](https://huggingface.co/models)下载预训练权重

## 开始训练
请执行
```bash
bash test/run_xxx.sh
```
拉起训练任务

## 训练结果

| Architecture  | Pretrained Model                                                                      | Script                                                                                                                              | eval | 
|---------------|---------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|------|
| sew           | [sew-small-100k](https://huggingface.co/asapp/sew-small-100k)                         | [run_ctc_sew.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/speech-recognition/run_ctc_sew.sh)                     | ✔️   |
| sew_d         | [sew-d-tiny-100k](https://huggingface.co/asapp/sew-d-tiny-100k)                       | [run_ctc_sew_d.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/speech-recognition/run_ctc_sew_d.sh)                 | ✔️   |
| unispeech     | [unispeech-large-1500h-cv](https://huggingface.co/microsoft/unispeech-large-1500h-cv) | [run_ctc_unispeech.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/speech-recognition/run_ctc_unispeech.sh)         | ✔️   |
| unispeech_sat | [unispeech-sat-base-plus](https://huggingface.co/microsfoft/unispeech-sat-base-plus)  | [run_ctc_unispeech_sat.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/speech-recognition/run_ctc_unispeech_sat.sh) | ✔️   |
| wavlm         | [wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus)                   | [run_ctc_wavlm.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/speech-recognition/run_ctc_wavlm.sh)                 | ✔️   |
| wav2vec2      | [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)                        | [run_ctc_wav2vec2.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/speech-recognition/run_ctc_wav2vec2.sh)           | ✔️   |
| whisper       | [whisper-tiny](https://huggingface.co/openai/whisper-tiny)                            | [run_seq2seq_whisper.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/speech-recognition/run_seq2seq_whisper.sh)     | ➖    |


## 遗留问题
- `run_seq2seq_whisper.sh` 数据集下载失败（网络问题），还未执行