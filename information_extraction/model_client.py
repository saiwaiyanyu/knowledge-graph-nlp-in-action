# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/11/10 4:46 PM
# @Author: wuchenglong

import grpc,json,collections
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from bert import tokenization
import numpy as np
from run_classifier import SkeProcessor
from run_classifier import _truncate_seq_pair


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


processor = SkeProcessor()

class SkePrediction(Prediction):

    def _config_(self,model):
        self.tokenizer = tokenization.FullTokenizer(vocab_file='chinese_L-12_H-768_A-12/vocab.txt',
                                                    do_lower_case = True)
        self.model_name = model
        self.signature_name = "serving_default"
        self.url = "localhost:8500"
        self.time_out = 10
        self.tag_to_id = dict(zip(processor.get_labels(), range(len(processor.get_labels()))))
        self.id_to_tag = dict(zip(range(len(processor.get_labels())), processor.get_labels()))
        self.max_seq_length= 128
        self.label_length= len(processor.get_labels())

    def update_input(self, text_list):
        self.text_list = text_list

    def input_process(self):
        input_ids = []
        input_mask = []
        segment_ids = []
        label_ids = []

        for text in self.text_list:
            input_info = text.split("\t")
            text_a = tokenization.convert_to_unicode(input_info[2])
            text_b = tokenization.convert_to_unicode(input_info[0] + input_info[1])

            tokens_a = self.tokenizer.tokenize(text_a)
            tokens_b = None
            if text_b:
                tokens_b = self.tokenizer.tokenize(text_b)

            if tokens_b:
                _truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)
            else:
                if len(tokens_a) > self.max_seq_length - 2:
                    tokens_a = tokens_a[0:(self.max_seq_length - 2)]

            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            if tokens_b:
                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
            label_ids = [0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                label_ids.append(0)

            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length
            assert len(label_ids) == self.max_seq_length

        print("*"*20)
        input = {"input_ids": [input_ids],
                 "segment_ids": [segment_ids],
                 "input_mask": [input_mask],
                 "label_ids": [label_ids],
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
            tf.contrib.util.make_tensor_proto(input["input_ids"],
                                              shape=[len(input["input_ids"]), self.max_seq_length])
        )
        request.inputs["segment_ids"].CopyFrom(
            tf.contrib.util.make_tensor_proto(input["segment_ids"],
                                              shape=[len(input["input_ids"]), self.max_seq_length])
        )
        request.inputs["input_mask"].CopyFrom(
            tf.contrib.util.make_tensor_proto(input["input_mask"],
                                              shape=[len(input["input_ids"]), self.max_seq_length])
        )
        request.inputs["label_ids"].CopyFrom(
            tf.contrib.util.make_tensor_proto(input["label_ids"],
                                              shape=[len(input["input_ids"]), self.max_seq_length])
        )

        response = stub.Predict(request, self.time_out)
        print("model version {}".format(response.model_spec.version.value))
        pre_path_list = np.array(response.outputs["probabilities"].float_val).reshape((-1, self.label_length))
        return pre_path_list

    def result(self):
        pre_path_list = self.predict()
        text_list = [ elem for elem in self.text_list]
        result_list = []
        for text, path in list(zip(text_list,pre_path_list)):
            result_list.append( self._format_result(text,path))
        return result_list

    def _format_result(self,chars,path):
        result =  [(self.id_to_tag[i], line)  for (i, line) in enumerate(list( path))]
        return sorted(result, key = lambda a:a[1], reverse=True)


if __name__=="__main__":

    model = "ske"

    predictions = {
        "ske": SkePrediction,
    }
    prediction = predictions[model]()
    prediction._config_(model)

    example =  "愤怒的唐僧	北京吴意波影视文化工作室	《愤怒的唐僧》由北京吴意波影视文化工作室与优酷电视剧频道联合制作，故事以喜剧元素为主，讲述唐僧与佛祖打牌，得罪了佛祖，被踢下人间再渡九九八十一难的故事"

    while True:
        text = input("input text as word_a word_b text: ")
        if not text:
            print("text is empty, input again, for example : {text}".format(text = example))
        try:
            pred_data = [text]
            prediction.update_input(pred_data)
            pred_result = prediction.result()
            for text, result in zip(pred_data, pred_result):
                print("\ntext: {}".format(text))
                result_map = collections.OrderedDict()
                for label,prob in result[0:5]:
                    result_map[label] = prob
                print("result:\n{}".format(json.dumps(
                    result_map, indent= 4 ,ensure_ascii=False
                )))
        except Exception as e:
            print("something wrong {}, check please".format(e))

