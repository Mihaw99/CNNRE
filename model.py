#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.7.5

import math
import numpy as np
import mindspore
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.common.initializer import Initializer, _calculate_fan_in_and_fan_out, _assignment, initializer, XavierUniform

class XavierNormal(Initializer):
    def __init__(self, gain=1):
        super().__init__(gain=gain)
        self.gain = gain

    def _initialize(self, arr):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(arr.shape)

        std = self.gain * math.sqrt(2.0 / float(fan_in + fan_out))
        data = np.random.normal(0, std, arr.shape)

        _assignment(arr, data)


class CNN(nn.Cell):
    def __init__(self, word_vec, class_num, config):
        super(CNN, self).__init__()
        self.word_vec = word_vec
        self.class_num = class_num

        # hyper parameters and others
        self.max_len = config.max_len
        self.word_dim = config.word_dim
        self.pos_dim = config.pos_dim
        self.pos_dis = config.pos_dis

        self.dropout_value = config.dropout
        self.filter_num = config.filter_num
        self.window = config.window
        self.hidden_size = config.hidden_size

        self.dim = self.word_dim + 2 * self.pos_dim

        # net structures and operations
        # self.word_embedding = nn.Embedding.from_pretrained(
        #     embeddings=self.word_vec,
        #     freeze=False,
        # )
        self.word_embedding = nn.Embedding(
            vocab_size=self.word_vec.shape[0],
            embedding_size=self.word_vec.shape[1],
        )

        # self.pos1_embedding = nn.Embedding(
        #     num_embeddings=2 * self.pos_dis + 3,
        #     embedding_dim=self.pos_dim
        # )
        self.pos1_embedding = nn.Embedding(
            vocab_size=2 * self.pos_dis + 3,
            embedding_size=self.pos_dim
        )
        # self.pos2_embedding = nn.Embedding(
        #     num_embeddings=2 * self.pos_dis + 3,
        #     embedding_dim=self.pos_dim
        # )
        self.pos2_embedding = nn.Embedding(
            vocab_size=2 * self.pos_dis + 3,
            embedding_size=self.pos_dim
        )

        # self.conv = nn.Conv2d(
        #     in_channels=1,
        #     out_channels=self.filter_num,
        #     kernel_size=(self.window, self.dim),
        #     stride=(1, 1),
        #     bias=True,
        #     padding=(1, 0),  # same padding
        #     padding_mode='zeros'
        # )
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.filter_num,
            kernel_size=(self.window, self.dim),
            stride=(1, 1),
            has_bias=True,
            padding=(1, 1, 0, 0), # 上下左右的顺序，源代码(1,0)为(行数，列数)
            pad_mode='pad'
        )

        # self.maxpool = nn.MaxPool2d((self.max_len, 1))
        self.maxpool = nn.MaxPool2d((self.max_len, 1))
        # self.tanh = nn.Tanh()
        self.tanh = nn.Tanh()
        # self.dropout = nn.Dropout(self.dropout_value)
        self.dropout = nn.Dropout(1-self.dropout_value)
        # self.linear = nn.Linear(
        #     in_features=self.filter_num,
        #     out_features=self.hidden_size,
        #     bias=True
        # )
        self.linear = nn.Dense(
            in_channels=self.filter_num,
            out_channels=self.hidden_size,
            has_bias=True
        )
        # self.dense = nn.Linear(
        #     in_features=self.hidden_size,
        #     out_features=self.class_num,
        #     bias=True
        # )
        self.dense = nn.Dense(
            in_channels=self.hidden_size,
            out_channels=self.class_num,
            has_bias=True
        )

        # initialize weight
        # init.xavier_normal_(self.pos1_embedding.weight)
        # init.xavier_normal_(self.pos2_embedding.weight)
        # init.xavier_normal_(self.conv.weight)
        # init.constant_(self.conv.bias, 0.)
        # init.xavier_normal_(self.linear.weight)
        # init.constant_(self.linear.bias, 0.)
        # init.xavier_normal_(self.dense.weight)
        # init.constant_(self.dense.bias, 0.)


        # print("embedding_table = ", self.pos1_embedding.embedding_table.asnumpy())
        self.pos1_embedding.embedding_table = initializer(XavierNormal(), self.pos1_embedding.embedding_table.shape)
        self.pos2_embedding.embedding_table = initializer(XavierNormal(), self.pos2_embedding.embedding_table.shape)
        self.conv.weight = initializer(XavierNormal(), self.conv.weight.shape)
        self.conv.bias = initializer(0, self.conv.bias.shape, mindspore.float32)
        self.linear.weight = initializer(XavierNormal(), self.linear.weight.shape)
        self.linear.bias = initializer(0, self.linear.bias.shape, mindspore.float32)
        self.dense.weight = initializer(XavierNormal(), self.dense.weight.shape)
        self.dense.bias = initializer(0, self.dense.bias.shape, mindspore.float32)

        self.concat = ops.Concat(axis=-1)


    def encoder_layer(self, token, pos1, pos2):
        # word_emb = self.word_embedding(token)  # B*L*word_dim
        # pos1_emb = self.pos1_embedding(pos1)  # B*L*pos_dim
        # pos2_emb = self.pos2_embedding(pos2)  # B*L*pos_dim
        # emb = torch.cat(tensors=[word_emb, pos1_emb, pos2_emb], dim=-1)
        # return emb  # B*L*D, D=word_dim+2*pos_dim

        word_emb = self.word_embedding(token)  # B*L*word_dim
        pos1_emb = self.pos1_embedding(pos1)  # B*L*pos_dim
        pos2_emb = self.pos2_embedding(pos2)  # B*L*pos_dim
        emb = self.concat([word_emb, pos1_emb, pos2_emb])
        return emb

    def conv_layer(self, emb, mask):
        # emb = emb.unsqueeze(dim=1)  # B*1*L*D
        # conv = self.conv(emb)  # B*C*L*1

        emb = ops.expand_dims(emb, axis=1)
        conv = self.conv(emb)

        # mask, remove the effect of 'PAD'
        # conv = conv.view(-1, self.filter_num, self.max_len)  # B*C*L
        # mask = mask.unsqueeze(dim=1)  # B*1*L
        # mask = mask.expand(-1, self.filter_num, -1)  # B*C*L
        # conv = conv.masked_fill_(mask.eq(0), float('-inf'))  # B*C*L
        # conv = conv.unsqueeze(dim=-1)  # B*C*L*1
        # return conv

        conv = conv.view(-1, self.filter_num, self.max_len)
        mask = ops.expand_dims(mask, axis=1)
        mask = ops.broadcast_to(mask, shape=(-1, self.filter_num, -1))
        conv = ops.masked_fill(conv, ops.equal(mask, 0), float('-inf'))
        conv = ops.expand_dims(conv, axis=-1)
        return conv

    def single_maxpool_layer(self, conv):
        pool = self.maxpool(conv)  # B*C*1*1
        pool = pool.view(-1, self.filter_num)  # B*C
        return pool

    def construct(self, data):
        token = data[:, 0, :].view(-1, self.max_len)
        pos1 = data[:, 1, :].view(-1, self.max_len)
        pos2 = data[:, 2, :].view(-1, self.max_len)
        mask = data[:, 3, :].view(-1, self.max_len)
        emb = self.encoder_layer(token, pos1, pos2)
        emb = self.dropout(emb)
        conv = self.conv_layer(emb, mask)
        pool = self.single_maxpool_layer(conv)
        sentence_feature = self.linear(pool)
        sentence_feature = self.tanh(sentence_feature)
        sentence_feature = self.dropout(sentence_feature)
        logits = self.dense(sentence_feature)
        return logits
