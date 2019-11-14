# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/11/10 4:46 PM
# @Author: wuchenglong


from my_log import logger
import tensorflow as tf
import shutil,json,os


def model_save(session, save_path, model_input, model_output):

    shutil.rmtree(save_path, True)
    builder = tf.saved_model.builder.SavedModelBuilder(save_path)

    # 定义输入签名
    inputs = {
        k: tf.saved_model.utils.build_tensor_info(v)
        for k, v in model_input.items()
    }

    # 定义输出签名
    outputs = {
        k:tf.saved_model.utils.build_tensor_info(v)
        for k, v in model_output.items()
    }

    # 创建签名对象
    signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs,
                                                                       outputs=outputs,
                                                                       method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                                                                       )
    # 将签名和标签加入到builder中
    builder.add_meta_graph_and_variables(session,
                                         [tf.saved_model.tag_constants.SERVING],
                                         {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
                                         )
    builder.save()
    logger.info("the latest model saved!!")

def get_tags(path, id_to_tag):
    return [id_to_tag[id]  for id in path]

def merge_two_dicts(dict_x, dict_y):
    """合并给定的2个dict"""
    dict_z = dict_x.copy()
    dict_z.update(dict_y)
    return dict_z

def obj_save(obj, json_path):
    if not os.path.exists(os.path.dirname(json_path)):
        os.makedirs(os.path.dirname(json_path))
    json.dump(obj, open(json_path, 'w'),indent=4)

def obj_load(json_path):
    para = json.load(open(json_path))
    return para

tag_check = {
    "I":["B","I"],
    "E":["B","I"],
}

def check_label(front_label,follow_label):
    if not follow_label:
        raise Exception("follow label should not both None")

    if not front_label:
        return True

    if follow_label.startswith("B-"):
        return False

    if (follow_label.startswith("I-") or follow_label.startswith("E-")) and \
        front_label.endswith(follow_label.split("-")[1]) and \
        front_label.split("-")[0] in tag_check[follow_label.split("-")[0]]:
        return True
    return False

def format_result(chars, tags):
    logger.info(tags)
    entities = []
    entity = []
    for index, (char, tag) in enumerate(zip(chars, tags)):
        entity_continue = check_label(tags[index - 1] if index > 0 else None, tag)
        if not entity_continue and entity:
            entities.append(entity)
            entity = []
        entity.append([index, char, tag, entity_continue])

    entities_result = []
    for entity in entities:
        if entity[0][2].startswith("B-"):
            entities_result.append(
                {"begin": entity[0][0],
                 "end": entity[-1][0],
                 "words": "".join([char for _, char, _, _ in entity]),
                 "type": entity[0][2].split("-")[1]
                 }
            )

    return entities_result

if __name__ == "__main__":
    text = ['[CLS]', '东', '方', '大', '气', '东', '方', '市', '国', '土', '局', '会', '同', '大', '田', '镇', '政', '府', '、', '市', '生', '态', '环', '境', '保', '护', '局', '、', '市', '交', '通', '运', '输', '局', '、', '市', '交', '警', '大', '队', '组', '[SEP]']
    tags =  ['[CLS]', 'B-TIM', 'I-TIM', 'I-ORG', 'O', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', '[SEP]']
    entities_result= format_result(text,tags)
    print(json.dumps(entities_result, indent=4, ensure_ascii=False))
