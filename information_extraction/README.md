
## 基于 Bert 的信息抽取(information extraction)，关系提取 (relation extraction)

## 环境依赖

 - redhat 7,centos 7,macOS 10.14
 - tensorflow 1.14.0
 - python 3.5.2


## 安装依赖
        
    cd information_extraction
    pip3 install -r requirement.txt
    # 若下载速度慢，可以选择阿里镜像
    # pip3 install -r requirement.txt -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

## 下载bert预训练模型

    $ cd information_extraction
    $ wget -c https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
    $ unzip chinese_L-12_H-768_A-12.zip 
    
    $ tree chinese_L-12_H-768_A-12
    bert_model
    ├── bert_config.json
    ├── bert_model.ckpt.data-00000-of-00001
    ├── bert_model.ckpt.index
    ├── bert_model.ckpt.meta
    └── vocab.txt

## 下载bert源代码
下载 [**bert**](https://github.com/google-research/bert) 放入项目目录**bert**下，

    $ cd information_extraction
    $ git clone https://github.com/google-research/bert.git
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
    ├── ske
    │   ├── dev.tsv
    │   └── train.tsv
        
### 数据格式

    $ head data/ske/train.tsv -n 2
    主演	喜剧之王	周星驰	如何演好自己的角色，请读《演员自我修养》《喜剧之王》周星驰崛起于穷困潦倒之中的独门秘笈
    UNKNOWN	周星驰	喜剧之王	如何演好自己的角色，请读《演员自我修养》《喜剧之王》周星驰崛起于穷困潦倒之中的独门秘笈    
    
说明：原始数据来源 [2019语言与智能技术竞赛](http://lic2019.ccf.org.cn/kg)。  官方数据共涵盖 50 种关系类型。根据原始数据，进行清洗整理，生成模型所需训练数据：
    
- **train.tsv 708231 条记录**
- **dev.tsv 88648 条记录**

因官方不再提供下载，如需数据用于学术研究等，可以联系 wuchenglong126@126.com，若用于商业用途，请联系竞赛主办方。

## 运行
#### 训练
    
    $ cd information_extraction
    $ export BERT_BASE_DIR=chinese_L-12_H-768_A-12
    $ export TASK_NAME=ske
    $ export OUTPUT_DIR=result
    $ export DATA_DIR=data
    
    $ python3 run_classifier.py \
      --task_name=ske \
      --do_train=true \
      --do_eval=true \
      --do_predict=true \
      --do_export=true \
      --data_dir=$DATA_DIR/$TASK_NAME \
      --vocab_file=$BERT_BASE_DIR/vocab.txt \
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \
      --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
      --max_seq_length=128 \
      --train_batch_size=32 \
      --learning_rate=2e-5 \
      --num_train_epochs=3.0 \
      --output_dir=$OUTPUT_DIR/$TASK_NAME/ckpt_model \
      --export_dir=$OUTPUT_DIR/$TASK_NAME/saved_model
    

### 训练结果

**result** 路径保存模型的训练结果，ckpt_model 主要为模型重新训练加载使用。saved_model 主要为savedModel 保存，推荐用于 **tensorflow_serving**  生产部署。

     $ tree  result/
    result/
    └── ske
        ├── ckpt_model
        │   ├── checkpoint
        │   ├── events.out.tfevents.1573897383.localhost.localdomain
        │   ├── events.out.tfevents.1573897778.localhost.localdomain
        │   ├── graph.pbtxt
        │   ├── model.ckpt-8000.data-00000-of-00001
        │   ├── model.ckpt-8000.index
        │   ├── model.ckpt-8000.meta
        │   ├── predict.tf_record
        ├── saved_model
        │   ├── 1573910572
        │   │   ├── saved_model.pb
        │   │   └── variables
        │   │       ├── variables.data-00000-of-00001
        │   │       └── variables.index
        ├── test_results.tsv
        └── train.tf_record


## serving 部署

    $ cd information_extraction
    
    $ BASE_DIR="$(pwd)"
    
    $ docker run  -t --rm  \
        -p 8500:8500 \
        -p 8501:8501 \
        --mount  type=bind,source=$BASE_DIR/result/ske/saved_model,target=/models/ske \
        -e MODEL_NAME=ske \
        -t tensorflow/serving
    

serving 服务调用(**grpc**)：

    $ python3 model_client.py

结果：

```
input: 喜剧之王      周星驰  如何演好自己的角色，请读《演员自我修养》《喜剧之王》周星驰崛起于穷困潦倒之中的独门秘笈

model version 1573956875

text: 喜剧之王  周星驰  如何演好自己的角色，请读《演员自我修养》《喜剧之王》周星驰崛起于穷困潦倒之中的独门秘笈
result:
{
    "主演": 0.9959173798561096,
    "导演": 0.0018050138605758548,
    "编剧": 0.000452475156635046,
    "歌手": 0.0002099768607877195,
    "制片人": 0.0001938332716235891,
    ...
}


```



