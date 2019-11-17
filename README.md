
## knowledge-graph-nlp-in-action
实战知识图谱和nlp相关任务，包括模型训练到部署全流程。

## 目录

* [NLP](#NLP)
    * [sequence labeling](#sequence-labeling)
    * [information extraction](#information-extraction)
* KG
    * [待补充]

    
## NLP

### [sequence labeling](./sequence_labeling)
**模型：** Bert + BiLSTM + CRF
**模型输入：** 一段文本。
**模型输出：** 文本包含的机构、人名、时间等实体。
    
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

[>>>>>>详情<<<<<<](./sequence_labeling)

### [information extraction](./information_extraction)

**模型：** Bert
**模型输入：** 实体A，实体B，包含实体A和实体B的文本。
**模型输出：** 文本包含的机构、人名、时间等实体。

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
    
[>>>>>>详情<<<<<<](./information_extraction)

