## UT测试用例使用

在/diffusers/example下各task目录下，存在文件test_examples_XXX, 该文件为对应task的UT测试用例

### 模型下载
如果代码无法自动从Huggingface下载模型，可从下面链接直接手动下载

| 模型权重                               | 链接    |
|------------------------------------|-------|
| hf-internal-testing/tiny-stable-diffusion-pipe | https://huggingface.co/hf-internal-testing/tiny-stable-diffusion-pipe |
| hf-internal-testing/tiny-stable-diffusion-xl-pipe | https://huggingface.co/hf-internal-testing/tiny-stable-diffusion-xl-pipe |
| hf-internal-testing/tiny-controlnet |https://huggingface.co/hf-internal-testing/tiny-controlnet|
| hf-internal-testing/tiny-controlnet-sdxl |https://huggingface.co/hf-internal-testing/tiny-controlnet-sdxl|
| hf-internal-testing/tiny-adapter   |https://huggingface.co/hf-internal-testing/tiny-adapter|

### 数据集下载
如果代码无法自动从Huggingface数据集，可从下面链接直接手动下载

| 任务                                     | 数据集                                           | 链接    |
|----------------------------------------|-----------------------------------------------|-------|
| controlnet <br> t2i_adapter            | hf-internal-testing/fill10 |https://huggingface.co/datasets/hf-internal-testing/fill10|
| custom_diffusion <br>textual_inversion | diffusers/cat_toy_example                     |https://huggingface.co/datasets/diffusers/cat_toy_example|
| dreambooth                             | diffusers/dog-example                         |https://huggingface.co/datasets/diffusers/dog-example|
| instruct_pix2pix                       | hf-internal-testing/instructpix2pix-10-samples |https://huggingface.co/datasets/hf-internal-testing/instructpix2pix-10-samples|
| text_to_image                          | lambdalabs/pokemon-blip-captions |https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions|
| unconditional_image_generation         | huggan/flowers-102-categories |https://huggingface.co/datasets/huggan/flowers-102-categories|


### accelerate配置
UT文件中自动配置了accelerate文件，这行代码为Huggingface源码，会自动读取机器配置，多卡设置为多卡，单机单卡会自动设置为单机单卡

```python
write_basic_config(save_location=cls.configPath)
```

如果多卡配置想要单机运行，可在UT文件中替换代码如下

```python
#write_basic_config(save_location=cls.configPath)
#cls._launch_args = ["accelerate", "launch", "--config_file", cls.configPath]
cls._launch_args = ["python3"]
```

### 运行脚本

- 直接运行全部UT case, 修改UT文件里main下函数如下，然后外部直接`python3`运行

```python
unittest.main()
```

- 运行其中一个case，修改UT文件里main下函数如下，，然后外部直接`python3`运行

```python
unittest.main(argv=[' ','ExamplesTestsAccelerate.test_train_unconditional'])
```

- 需要保存日志的运行方式
```shell
python3 test_examples_XXX >> test_examples.log 2>&1
```

### 测试注意事项

基于参数`checkpointing_steps`和`checkpoints_total_limit`，diffusers支持保存对应steps下的权重结果。这两个参数基于`max_train_steps`提供的steps进行保存，而Huggingface提供的源码里计算`max_train_steps`方式并不统一，同时NPU数目的不同也会影响`max_train_steps`,如果UT测试时碰到涉及checkpoint保存的用例，可以查看日志输出的结果，进而修正`max_train_steps`。