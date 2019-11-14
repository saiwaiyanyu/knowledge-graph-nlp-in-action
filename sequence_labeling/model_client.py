# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/11/10 4:46 PM
# @Author: wuchenglong

import grpc,json
import tensorflow as tf
import tensorflow.keras as kr
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tokenization
import helper
import numpy as np


class Prediction(object):

    def __init__(self):
        pass

    def _config_(self,model):
        """模型调用的一些配置信息"""
        pass

    def input_process(self):
        """对输入信息进行预处理"""
        pass

    def predict(self ):
        """调用模型预测"""
        pass

    def path_id_to_tag(self, path):
        """对路径转换成tag"""
        pass

    def result(self):
        """对模型结果机构化，返回预测结果"""
        pass

    def _format_result(self,chars,path):
        """一些模型结构化的处理"""
        pass


class NerPrediction(Prediction):

    def _config_(self,model):
        self.tokenizer = tokenization.FullTokenizer(vocab_file='result/{model}/saved_model/vocab.txt'.format(model=model))
        self.model_name = model
        self.signature_name = "serving_default"
        self.url = "127.0.0.1:8500"
        self.time_out = 10
        self.id_to_tag = helper.obj_load("result/{model}/saved_model/id_to_tag.json".format(model=model))

    def update_input(self, text_list):
        self.text_list = text_list

    def input_process(self):
        input_ids = []
        input_mask = []
        segment_ids = []

        for text in self.text_list:
            tokens = ["[CLS]"] + list(text) + ["[SEP]"]
            input_ids.append(self.tokenizer.convert_tokens_to_ids(tokens))
            input_mask.append([1] * len(tokens))
            segment_ids.append([0] * len(tokens))

        self.max_length = max( [len(elem)  for elem in input_ids])

        # 使用pad_sequences来将文本pad为固定长度
        input_ids = kr.preprocessing.sequence.pad_sequences(input_ids, self.max_length, padding="post", truncating="post")
        input_mask = kr.preprocessing.sequence.pad_sequences(input_mask, self.max_length, padding="post", truncating="post")
        segment_ids = kr.preprocessing.sequence.pad_sequences(segment_ids, self.max_length, padding="post",truncating="post")

        input = {"input_ids": input_ids,
                 "segment_ids": segment_ids,
                 "input_mask": input_mask,
                 "dropout": 1.0
                 }
        return input

    def path_id_to_tag(self,path):
        return [self.id_to_tag[str(elem)] for elem in path]

    def predict(self):
        input = self.input_process()
        print("predict...")
        channel = grpc.insecure_channel(self.url)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        # request.model_spec.version.value = 20191113
        request.model_spec.signature_name = self.signature_name
        request.inputs["input_ids"].CopyFrom(
            tf.contrib.util.make_tensor_proto(input["input_ids"], shape=[len(input["input_ids"]), self.max_length]))
        request.inputs["segment_ids"].CopyFrom(
            tf.contrib.util.make_tensor_proto(input["segment_ids"], shape=[len(input["input_ids"]), self.max_length]))
        request.inputs["input_mask"].CopyFrom(
            tf.contrib.util.make_tensor_proto(input["input_mask"], shape=[len(input["input_ids"]), self.max_length]))
        request.inputs['dropout'].CopyFrom(
            tf.contrib.util.make_tensor_proto(input["dropout"]))

        response = stub.Predict(request, self.time_out)
        # print(response.model_spec.version.value)
        # result = stub.Predict(request, time_out).outputs["pre_paths"].int_val  # 10 secs timeout
        pre_path_list = np.array(response.outputs["pre_paths"].int_val).reshape((-1, self.max_length))
        return pre_path_list

    def result(self):
        pre_path_list = self.predict()
        text_list = [ ["[CLS]"] + list(elem) + ["[SEP]"]   for elem in self.text_list]
        result_list = []
        for text, path in list(zip(text_list,pre_path_list)):
            result_list.append( self._format_result(text,path))
        return result_list

    def _format_result(self,chars,path):
        tags = self.path_id_to_tag(path)
        return helper.format_result(chars, tags)


if __name__=="__main__":

    model = "ner"

    predictions = {
        "ner": NerPrediction,
    }
    prediction = predictions[model]()
    prediction._config_(model)

    example =  " 塑造了毛泽东、周恩来、刘少奇、朱德、陈云"

    while True:
        text = input("input text : ")
        if not text:
            print("text is empty, input again, for example : {text}".format(text = example))
        pred_data = [text]
        prediction.update_input(pred_data)
        pred_result = prediction.result()
        for text, entity in zip(pred_data, pred_result):
            print("\ntext: {}".format(text))
            print("\nresult:\n{}".format(json.dumps(entity, indent=4, ensure_ascii=False)))

