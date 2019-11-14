
## 基于 Bert + Bi-LSTM + CRF 的命名实体识别(TensorFlow)

## 环境依赖

 - redhat 7,centos 7,macOS 10.14
 - tensorflow 1.14.0
 - python 3.5.2


## 安装依赖
        
    cd sequence_labeling
    pip3 install -r requirement.txt
    # 若下载速度慢，可以选择阿里镜像
    # pip3 install -r requirement.txt -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

## 下载bert预训练模型

    $ cd sequence_labeling
    $ wget -c https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
    
    #解压后放到项目 **bert_model** 下
    $ unzip chinese_L-12_H-768_A-12.zip && mv chinese_L-12_H-768_A-12 bert_model
    
    $ tree bert_model
    bert_model
    ├── bert_config.json
    ├── bert_model.ckpt.data-00000-of-00001
    ├── bert_model.ckpt.index
    ├── bert_model.ckpt.meta
    └── vocab.txt

## 下载bert源代码
下载 [**bert**](https://github.com/google-research/bert) 放入项目目录**bert**下，

    $ cd sequence_labeling
    $ git clone https://github.com/google-research/bert.git bert
    $ tree bert
    bert
    ├── CONTRIBUTING.md
    ├── create_pretraining_data.py
    ├── extract_features.py
    ├── __init__.py
    ├── LICENSE
    ├── modeling.py
    ├── modeling_test.py
    ├── multilingual.md
    ├── optimization.py
    ├── optimization_test.py
    ├── predicting_movie_reviews_with_bert_on_tf_hub.ipynb
    ├── requirements.txt
    ├── run_classifier.py
    ├── run_classifier_with_tfhub.py
    ├── run_pretraining.py
    ├── run_squad.py
    ├── sample_text.txt
    ├── tokenization.py
    └── tokenization_test.py
    
## 数据

    $ tree data
    data
    ├── ner
    │   ├── dev
    │   └── train
    
### 数据格式


    $ head data/ner/train -n 20
    
    迈      O
    向      O
    充      O
    满      O
    希      O
    望      O
    的      O
    新      O
    世      O
    纪      O
    —       O
    —       O
    一      B_TIME
    九      M_TIME
    九      M_TIME
    八      M_TIME
    年      E_TIME
    新      B_TIME
    年      E_TIME
    讲      O
    

## 项目结构

|  路径| 说明 | 
| --- | --- | 
|./bert| bert google 官方模型|
|./bert_model|bert 中文预训练模型|
|./data||
|./data/ner||
|./data/ner/dev|验证数据|
|./data/ner/train|训练数据|
|./data_processor.py|数据预处理脚本|
|./helper.py|相关模型函数|
|./log|日志记录|
|./model_client.py|调用模型client客户端|
|./model.py|主要模型|
|./my_log.py|日志模块|
|./requirement.txt|项目模型依赖|
|./result||
|./result/ner||
|./result/ner/ckpt_model| ckpt 模型结果|
|./result/ner/saved_model|savedModel模型结果|
|./tokenization.py|对token的处理脚本|

## 运行
#### 训练
    
    $ cd sequence_labeling
    $ python3 model.py -t ner -e train
    
![](media/15736923637849.jpg)

### 训练结果

**result** 路径保存模型的训练结果，ckpt_model 主要为模型重新训练加载使用。saved_model 主要为savedModel 保存，推荐用于 **tensorflow_serving**  生产部署。

     $ tree  result/
    result/
    └── ner
        ├── ckpt_model
        │   └── 20191113
        │       ├── checkpoint
        │       ├── ner.org.ckpt.data-00000-of-00001
        │       ├── ner.org.ckpt.index
        │       └── ner.org.ckpt.meta
        └── saved_model
            ├── 20191113
            │   ├── saved_model.pb
            │   └── variables
            │       ├── variables.data-00000-of-00001
            │       └── variables.index
            ├── id_to_tag.json
            ├── parameter_information.json
            ├── tag_to_id.json
            └── vocab.txt

### 测试

    $ python3 model.py -t ner -e predict

```json

input text : 中共中央总书记、国家主席江泽民发表１９９８年新年讲话
[
    {
        "end": 4,
        "words": "中共中央",
        "type": "ORG",
        "begin": 1
    },
    {
        "end": 15,
        "words": "江泽民",
        "type": "PER",
        "begin": 13
    },
    {
        "end": 22,
        "words": "１９９８年",
        "type": "TIME",
        "begin": 18
    }
]

```

## serving 部署

    $ cd sequence_labeling
    
    $ BASE_DIR="$(pwd)"
    
    $ docker run  -t --rm  \
        -p 8500:8500 \
        -p 8501:8501 \
        --mount  type=bind,source=$BASE_DIR/result/ner/saved_model,target=/models/ner \
        -e MODEL_NAME=ner \
        -t tensorflow/serving
    

serving 服务调用(**grpc**)：

    $ python3 model_client.py


![](media/15736926131208.jpg)



