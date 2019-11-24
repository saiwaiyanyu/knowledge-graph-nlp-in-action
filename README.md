
## knowledge-graph-nlp-in-action
实战知识图谱和nlp相关任务，包括模型训练到部署全流程。

## 目录

* [NLP](#NLP)
    * [x] [sequence labeling](#sequence-labeling)
    * [x] [information extraction](#information-extraction)
    * [x] [seq2seq](#seq2seq)
* KG
    * [ ] TODO

    
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
    
> ***** Eval results *****
eval_accuracy = 0.97999
eval_loss = 0.06774125
global_step = 19000
loss = 0.06772543



### [seq2seq](./seq2seq)

**模型：** seq2seq greedy （QA）

**模型输入：** 输入你的梦境。

**模型输出：** 返回梦境解析的结果（**周公解梦**数据训练）。

    ********************
    input your dream: 梦见中奖了
    dream: 梦见中奖了
    dream decoding: 预示你事业上将面临挑战和机遇，会有大发展。 
    
    ********************
    input your dream: 梦见大富豪
    dream: 梦见大富豪
    dream decoding: 预示着自己生活会很愉快。
    
## 相关文档

- [x] [`docker` 安装](./docs/docker安装.md)
- [ ] [`savedModel` 保存和 `tensorflow-serving` 部署](./docs/savedModel保存和tensorflow-serving部署.md)
- [ ] [`python` 和 `java` 调用 `tensorflow-serving` 服务](./docs/python和java调用tensorflow-serving服务.md)

