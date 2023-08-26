# Speech recognition

本项目展示如何在Ascend NPU下运行Tansformers的[speech-recognition](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition)任务。

## 准备训练环境
### 准备环境
- 环境准备指导。
  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  当前基于 PyTorch 2.1.0 完成测试。
- 安装依赖
  
  1、使用 NPU 设备源码安装Huggingface的Github仓的 accelerate
  ```text
  git clone https://github.com/huggingface/accelerate
  cd accelerate
  pip3 install -e .
  ```
  2、使用 NPU 设备源码安装 Transformers 插件
  ```text
  git clone https://github.com/huggingface/transformers
  cd transformers
  pip3 install -e .
  ```

  3、安装question_answering任务所需依赖
  ```text
  pip3 install -r requirements.txt
  ```

### 准备数据集
数据集下载地址：
```text
  https://huggingface.co/datasets
```

### 准备预训练权重
预训练权重类路径:https://huggingface.co/models
请按需下载相应的权重并置于`run_speech_recognition_ctc.py`同级目录下。

## 开始训练
执行
```bash
bash ./scripts/run_XXX.sh
```

## 训练结果

| Architecture | Pretrained Model                                                                 | Script                                                                                                                    | Device | Performance(8-cards) | Wer    |
|--------------|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|--------|----------------------|--------|
| wav2vec2     | [wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) | [run_wav2vec2.sh](https://gitee.com/ascend/transformers/tree/develop/examples/speech-recognition/scripts/run_wav2vec2.sh) | 910B   | 7                    | 0.6842 |


### FAQ
- 权重例如wav2vec2-large-xlsr-53文件夹要放在与执行脚本run_speech_recognition_ctc.py相同的路径下
- libsndfile与soundfile相关问题的说明
  如遇以下报错：
  OSError: cannot load library 'libsndfile.so': libsndfile.so: cannot open shared object file: No such file or directory
  或
  soundfile.LibsndfileError: Error opening ‘.mp3‘: File contains data in an unknown format.
  
  解决方案：
  首先排查是否为数据集问题，确认无误后，下载[soundfile](https://github.com/bastibe/python-soundfile)
  按照README中的Building步骤进行安装。
  安装完成后，cd到_soundfile_data目录下，根据机器的是，x86还是arm，选择对应的.so文件，将名称修改为libsndfile.so，记下libsndfile.so所在的路径，假设为path-to-libsndfile（不包含libsndfile.so）
  在终端输入：
  ```
  export LD_LIBRARY_PATH=path-to-libsndfile:$LD_LIBRARY_PATH
  ```
  若无法解决问题，则源码编译libsndfile：
  下载libsndfile源码仓:
  ```
  git clone https://github.com/bastibe/libsndfile-binaries.git
  ```
  按照如下进行编译安装：
  ```
  conda install -c conda-forge automake libogg libtool mpg123 pkg-config libopus libvorbis libflac
  yum install autogen
  bash linux_build.sh
  cp /libsndfile.so <your_lib_path> 
  ```
  编译结束后，在终端输入以下指令：
  export LD_PRELOAD=<your_lib_path>/libsndfile.so:$LD_PRELOAD
  export LD_LIBRARY_PATH=<your_lib_path>:$LD_LIBRARY_PATH
  


