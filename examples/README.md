# Examples
本目录包含原生 Transformers 套件官方用例，这些用例按照 NLP 任务划分。
本插件对官方用例的支持情况如下表所示。

| 任务                                                   | 数据集       | 是否基于Tranier | 晟腾NPU单卡训练 | 晟腾NPU八卡训练 
|------------------------------------------------------|-----------|:-----------:|:---------:|:---------:|
| [**`question-answering`**](./question-answering)      | SQuAD |      ✅      |     ✅     |     ✅     | 
| [**`text-classification`**]()                     | GLUE      |      ✅      |     -     |     -     | 
| [**`language-modeling`**]()               | WikiText-2     |      ✅      |     -     |     -     | 
| [**`summarization`**]()                         | XSum      |      ✅      |     -     |     -     | 
| [**`multiple-choice`**]()             | SWAG      |      ✅      |     -     |     -     |
| [**`text-generation`**]()                     | -         |     n/a     |     -     |     -     | 
| [**`token-classification`**]()           | CoNLL NER |      ✅      |     -     |     -     |
| [**`translation`**]()                             | WMT       |      ✅      |     -     |     -     | 
| [**`speech-recognition`**]()               | TIMIT     |      ✅      |     -     |     -     | 
| [**`multi-lingual speech-recognition`**]() | Common Voice |      ✅      |     -     |     -     | 
| [**`audio-classification`**]()           | SUPERB KS |      ✅      |     -     |     -     | 
| [**`image-classification`**]()                | CIFAR-10  |      ✅      |     -     |     -     |

## 执行训练脚本
详具体任务下的README，如[问答任务](./question-answering/README.md)。