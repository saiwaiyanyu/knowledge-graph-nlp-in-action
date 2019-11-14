# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/11/10 4:46 PM
# @Author: wuchenglong

import json,sys,time,shutil, os
import helper
import tokenization
from bert import modeling
from bert.optimization import create_optimizer
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from my_log import logger
import data_processor
import argparse


parser = argparse.ArgumentParser(
    usage = "usage: python3 model.py -t ner -e train/predict "
)

parser.add_argument("-t", "--task", type=str, default="ner")
parser.add_argument("-e", "--entry", type=str, default="train")
ARGS = parser.parse_args()


class Model():

    def __init__(self, config):
        self.config = config
        self.task_name = config["task_name"]
        self.lstm_dim = config["lstm_dim"]
        self.embedding_size = config["embedding_size"]
        self.max_epoch = config["max_epoch"] ######原为10 epoch
        self.learning_rate = config["learning_rate"]
        self.checkpoint_dir = config["checkpoint_dir"]
        self.checkpoint_path = config["checkpoint_path"]
        self.initializer = initializers.xavier_initializer()
        self.is_training = True if ARGS.entry=="train" else False
        self.bert_config = config["bert_config"]
        self.init_checkpoint = config["init_checkpoint"]
        self.vocab_dir = config["vocab_dir"]
        self.tf_serving_save_dir = config["tf_serving_save_dir"]
        self.predict_file = config["predict_file"]
        self.predict_result = config["predict_result"]
        self.require_improvement = config["require_improvement"]
        self.global_steps = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_f1 = 0.0
        self.best_match_num = 0
        self.steps = 0 # 迭代次数
        self.last_improved = 0 # 记录上一次提升批次
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab_dir,
        )


    def creat_model(self):

        self._init_placeholder()

        self.add_bert_layer()

        self.add_biLSTM_layer()

        self.add_logits_layer()

        self.add_loss_layer()

        self.add_optimizer_layer()

    def _init_placeholder(self):
        self.input_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="input_ids"
        )
        self.input_mask = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="input_mask"
        )
        self.segment_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="segment_ids"
        )
        self.targets = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="targets"
        )
        self.dropout = tf.placeholder(
            dtype=tf.float32,
            shape=None,
            name="dropout"
        )
        used = tf.sign(tf.abs(self.input_ids))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.length = tf.cast(length, tf.int32)
        self.nums_steps = tf.shape(self.input_ids)[-1]

    def add_bert_layer(self):
        bert_config = modeling.BertConfig.from_json_file(self.bert_config)

        model = modeling.BertModel(
            config=bert_config,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False
        )
        self.embedded = model.get_sequence_output()
        self.model_inputs = tf.nn.dropout(self.embedded, self.dropout)

    def add_biLSTM_layer(self):
        with tf.variable_scope("bi-LSTM") as scope:
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = tf.contrib.rnn.LSTMCell(
                        num_units=self.lstm_dim,
                        # use_peepholes=True,
                        # initializer=self.initializer,
                        state_is_tuple=True
                    )

            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_cell['forward'],
                cell_bw=lstm_cell['backward'],
                inputs=self.model_inputs,
                sequence_length=self.length,
                dtype=tf.float32,
            )
            self.lstm_outputs = tf.concat(outputs, axis=2)

    def add_logits_layer(self):
        with tf.variable_scope("hidden"):
            w = tf.get_variable("W",
                                shape=[self.lstm_dim*2, self.lstm_dim],
                                dtype=tf.float32,
                                initializer=self.initializer
                                )
            b = tf.get_variable("b",
                                shape=[self.lstm_dim],
                                dtype=tf.float32,
                                initializer=self.initializer
                                )

            output = tf.reshape(self.lstm_outputs, shape=[-1, self.lstm_dim*2])
            self.hidden = tf.tanh( tf.matmul(output, w) + b)

        with tf.variable_scope("logits"):
            w = tf.get_variable("W",
                                     shape=[self.lstm_dim, self.nums_tags],
                                     initializer=self.initializer,
                                     dtype=tf.float32
                                )
            b = tf.get_variable("b", shape=[self.nums_tags], dtype=tf.float32)
            pred = tf.matmul(self.hidden, w) + b
            self.logits = tf.reshape(pred, shape=[-1, self.nums_steps, self.nums_tags])

    def add_loss_layer(self):
        with tf.variable_scope("loss_layer"):
            self.trans = tf.get_variable(
                "transitions",
                shape=[self.nums_tags, self.nums_tags],
                initializer=self.initializer
            )

            log_likelihood, self.trans = tf.contrib.crf.crf_log_likelihood(
                inputs=self.logits,
                tag_indices=self.targets,
                transition_params=self.trans,
                sequence_lengths=self.length
            )

            self.paths, _ = tf.contrib.crf.crf_decode(
                potentials = self.logits,
                transition_params = self.trans,
                sequence_length=  self.length
            )
            self.outputs = tf.identity(self.paths, name="outputs")
            self.loss = tf.reduce_mean(-log_likelihood)

    def add_optimizer_layer(self):
        correct_prediction = tf.equal(
            self.paths, tf.cast(self.targets, tf.int32)
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        num_train_steps = int(
            self.train_length / self.batch_size * self.max_epoch)
        num_warmup_steps = int(num_train_steps * 0.1)
        self.train_op = create_optimizer(
            self.loss, self.learning_rate, num_train_steps, num_warmup_steps, False
        )
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)


    def train_step(self, sess, batch):
        tokens, tag_ids, inputs_ids, segment_ids, input_mask = zip(*batch)
        feed = {
            self.input_ids: inputs_ids,
            self.targets: tag_ids,
            self.segment_ids: segment_ids,
            self.input_mask: input_mask,
            self.dropout: 0.5
        }

        embedding, global_steps, loss, _, logits, acc, length = sess.run(
            [self.embedded,
            self.global_steps,
            self.loss,
            self.train_op,
            self.logits,
            self.accuracy,
            self.length
             ], feed_dict=feed)
        return global_steps, loss, logits, acc, length

    def train(self):

        self.train_data = processors[self.task_name](self.tokenizer, "train", self.config)
        self.dev_data = processors[self.task_name](self.tokenizer, "dev", self.config)

        logger.info("---"*20)
        self.batch_size = self.train_data.batch_size
        self.nums_tags = len(self.train_data.get_tags())
        self.tag_to_id = self.train_data.tag_to_id
        self.id_to_tag = self.train_data.id_to_tag
        self.train_length = len(self.train_data.data)

        self.dev_batch = self.dev_data.iteration()
        self.dev_length = len(self.dev_data.data)

        logger.info("-"*50)
        logger.info("train data:\t %s", self.train_length)
        logger.info("dev data:\t %s", self.dev_length)
        logger.info("nums of tags:\t %s", self.nums_tags)
        logger.info("tag_to_id: {}".format(self.tag_to_id))
        logger.info("id_to_tag: {}".format(self.id_to_tag))

        self.creat_model()
        with tf.Session() as sess:
            with tf.device("/cpu:0"):
                # 是否加载训练模型
                if not os.path.exists(self.checkpoint_dir):
                    os.makedirs(self.checkpoint_dir)
                ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    logger.info("restore model")    # 加载预训练模型
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    sess.run(tf.global_variables_initializer())

                t_vars = tf.trainable_variables()
                (assignment_map, initialized_variable_names) = \
                    modeling.get_assignment_map_from_checkpoint(t_vars,
                                                             self.init_checkpoint)
                tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)

                self.model_input = {
                    "input_ids": self.input_ids,
                    "segment_ids": self.segment_ids,
                    "input_mask": self.input_mask,
                    "dropout": self.dropout
                }

                self.model_output = {
                    "logits": self.logits,
                    "length": self.length,
                    "pre_paths": self.paths
                }

                for i in range(self.max_epoch):
                    logger.info("-"*50)
                    logger.info("epoch {}".format(i))
                    self.steps = 0

                    for batch in self.train_data.get_batch():
                        self.steps += 1
                        global_steps, loss, logits, acc, length = self.train_step(sess, batch)

                        if self.steps % 1 == 0:
                            logger.info("[->] epoch {}: step {}/{}\tloss {:.4f}\tacc {:.5f}".format(
                                i,self.steps, len(self.train_data.batch_data), loss, acc))

                        if self.steps % 20 == 0:
                            self.evaluate(sess)

                        if self.steps - self.last_improved > self.require_improvement:
                            logger.warn("No optimization for a long time, auto-stopping...")
                            break
                logger.info("training finished!!!")

    def evaluate(self, sess):

        batch = self.dev_batch.__next__()
        tokens, tag_ids, inputs_ids, segment_ids, input_mask = zip(*batch)

        feed = {
            self.input_ids: inputs_ids,
            self.segment_ids: segment_ids,
            self.targets: tag_ids,
            self.input_mask: input_mask,
            self.dropout: 1.0
        }

        scores, acc, lengths, pre_paths = sess.run([self.logits, self.accuracy, self.length, self.paths], feed_dict=feed)

        logger.info(tokens[0])
        logger.info(inputs_ids[0])
        logger.info(tag_ids[0])
        logger.info(pre_paths[0])
        match_num = sum([1 if helper.get_tags(tar, self.id_to_tag) == helper.get_tags(pre, self.id_to_tag)  else 0 for tar, pre in zip(tag_ids, pre_paths) if any([elem for elem in tar if elem not in (0, 1, 2, 3)])])
        logger.info("\tmatch {:.4f}\t best_match_num {:.4f}".format(match_num,self.best_match_num))
        logger.info("\tacc {:.4f}\t best {:.4f}\t".format(acc, self.best_f1))
        if acc >= self.best_f1 or match_num >= self.best_match_num:
            logger.info("acc {:.4f}, match_num {:.4f}  saved model!!".format(acc,match_num))
            self.saver.save(sess, self.checkpoint_path)

            if acc > self.best_f1:
                self.best_f1 = acc
            if match_num > self.best_match_num:
                self.best_match_num = match_num
            self.last_improved = self.steps
            helper.model_save(session=sess,
                              save_path=self.tf_serving_save_dir,
                              model_input=self.model_input,
                              model_output=self.model_output
                              )

    def prepare_pred_data(self, text):
        max_length = len(text) + 2
        tokens = list(text)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        logger.info(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        input_ids = input_ids + (max_length - len(input_ids)) * [0]
        segment_ids = segment_ids + (max_length - len(segment_ids)) * [0]
        input_mask = input_mask + (max_length - len(input_mask)) * [0]

        feed = {
            self.input_ids: [input_ids],
            self.segment_ids: [segment_ids],
            self.input_mask: [input_mask],
            self.dropout: 1.0
        }
        return feed


    def predict(self):

        self.tag_to_id = helper.obj_load(self.config["tag_to_id"])
        self.id_to_tag = {int(id): tag for id, tag in helper.obj_load(self.config["id_to_tag"]).items()}
        self.batch_size = 1
        self.train_length = 10

        self.nums_tags = len(self.tag_to_id.keys())
        self.creat_model()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                logger.info("[->] restore model")
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                logger.info("[->] no model, error,please check cpk dir: {}".format(self.checkpoint_dir))
                return
                # sess.run(tf.global_variables_initializer())

            while True:
                text = input("input text : ")
                logger.info('text: %s', text)
                feed = self.prepare_pred_data(text)
                logits, length, paths = sess.run([self.logits, self.length, self.paths], feed_dict=feed)
                logger.info(['paths: ', paths])
                logger.info(['tag_map: ',self.id_to_tag])
                logger.info(["tag",[self.id_to_tag[elem] for elem in paths[0] ] ])
                entities_result =  helper.format_result(["[CLS]"] + list(text) + ["[SEP]"], [self.id_to_tag[elem] for elem in paths[0]])
                print(json.dumps(entities_result, indent=4, ensure_ascii=False))



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("arg error！！")
        exit()

    processors = {
        "alias": data_processor.NerProcessor,
        "ner": data_processor.NerProcessor,
    }
    task_name = ARGS.task
    if task_name not in processors:
        print("Task not found: %s" % (task_name))
        exit()

    if ARGS.entry == "train":

        para = {
            "lstm_dim": 128,
            "max_epoch": 40,
            "train_batch": 16,
            "dev_batch": 256,
            "require_improvement": 1000
        }

        logger.warn("--------" * 10)
        logger.warn("\npara : \n {para}".format(
            para=json.dumps(para, indent=4, ensure_ascii=False)))
        base_config = {"task_name": task_name,
                        "mode": "bert" ,
                        "lstm_dim": 128,
                        "embedding_size": 50,
                        "max_epoch": 10,
                        "train_batch": 16,
                        "dev_batch": 128,
                        "learning_rate": 5e-5,
                        "require_improvement": 500,
                        "bert_config": "bert_model/bert_config.json",
                        "init_checkpoint": "bert_model/bert_model.ckpt",
                        "vocab_dir": "bert_model/vocab.txt",
                        "checkpoint_dir": "./result/{task_name}/ckpt_model/{model_version}".format(task_name=task_name,model_version = time.strftime('%Y%m%d')),# %Y%m%d%H%M%S
                        "checkpoint_path": "./result/{task_name}/ckpt_model/{model_version}/ner.org.ckpt".format(task_name=task_name,model_version = time.strftime('%Y%m%d')),
                        "train_file": "data/{task_name}/train".format(task_name=task_name),
                        "dev_file": "data/{task_name}/dev".format(task_name=task_name),
                        "predict_file": "data/{task_name}/predict".format(task_name=task_name),
                        "predict_result": "data/{task_name}/predict_result".format(task_name=task_name),
                        "tf_serving_save_dir": "result/{task_name}/saved_model/{model_version}".format(task_name=task_name,model_version = time.strftime('%Y%m%d')),
                        "parameter_information": "result/{task_name}/saved_model/parameter_information.json".format(task_name=task_name),
                        "save_dir": "result/{task_name}/saved_model/".format(task_name=task_name),
                        "tag_to_id": "result/{task_name}/saved_model/tag_to_id.json".format(task_name=task_name),
                        "id_to_tag": "result/{task_name}/saved_model/id_to_tag.json".format(task_name=task_name)
                        }

        bert_config = helper.obj_load(base_config["bert_config"])
        base_config = helper.merge_two_dicts(base_config, para)
        config = {
            "base_config": base_config,
            "bert_config": bert_config
                }
        helper.obj_save(config, base_config["parameter_information"])
        if os.path.exists(os.path.join(base_config["save_dir"], "vocab.txt")):
            logger.debug(["model_result vocab_file existed!!"])
        else:
            shutil.copy(base_config["vocab_dir"], base_config["save_dir"])
        logger.info(base_config)
    else:
        base_config = helper.obj_load("result/{task_name}/saved_model/parameter_information.json".format(task_name=task_name))["base_config"]

    model = Model(base_config)

    if ARGS.entry == "train":
        model.train()
    elif ARGS.entry == "predict":
        model.predict()
