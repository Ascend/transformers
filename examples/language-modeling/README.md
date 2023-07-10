# Language model training

本项目展示如何在Ascend NPU下运行Tansformers的[language-modeling](https://github.com/huggingface/transformers/tree/v4.30.2/examples/pytorch/language-modeling)任务。

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
language-modeling示例使用的数据集由Hugging Face提供的接口自动下载无需额外准备，，如果出现数据集和metric预处理脚本下载失败的问题，请手动下载

- metric: 把[accuracy](https://github.com/huggingface/evaluate/tree/main/metrics/accuracy)放到`language-modeling`目录下
- datatsets: 把[wikitext](https://huggingface.co/datasets/wikitext)放到`language-modeling`目录下

```text
 language-modeling
    ├── accuracy
    │   ├── accuracy.py
    │   └── app.py
    ├── wikitext
    │   ├── dataset_infos.json
    │   └── wikitext.py
    └── ... 
```


### 准备预训练权重
请从[huggingface hub](https://huggingface.co/models)下载预训练权重

## 开始训练
请执行
```bash
bash test/run_xxx.sh
```
拉起训练任务

## 训练结果

| Architecture | Pretrained Model                                            | Script                                                                                                 | supported | 
|--------------|-------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|-----------|
| gpt2         | [gpt2](https://huggingface.co/gpt2)                         | [run_clm.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/language-modeling/run_clm.sh) | ✔️        |
| roberta      | [roberta-base](https://huggingface.co/roberta-base)         | [run_mlm.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/language-modeling/run_mlm.sh) | ✔️        |
| xlnet        | [xlnet-base-cased](https://huggingface.co/xlnet-base-cased) | [run_plm.sh](https://gitee.com/ascend/transformers/blob/v4.30.2/examples/language-modeling/run_plm.sh) | ✔️        |


## 遗留问题
- `run_plm.sh` 性能很差，`9.7s/it`，待定位
