# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import six
import numpy as np
import torch
import torch.nn as nn
from .transform import BERTLayerNorm


def gelu(inputs):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return inputs * 0.5 * (1.0 + torch.erf(inputs / math.sqrt(2.0)))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}


class WordEmbedding(nn.Module):
    ''' convert word sequence to embeddings '''

    def __init__(self, vocab_size, embed_dim, use_pretrained_embed=False,
                 pretrained_embed=None, fine_tune=True, **kwargs):
        super(WordEmbedding, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if use_pretrained_embed:
            pretrained_embed = torch.FloatTensor(pretrained_embed)  # .to(conf.device)
            self.embedding.weight.data.copy_(pretrained_embed)
        self.embedding.weight.requires_grad = fine_tune

    def forward(self, inputs):
        return self.embedding(inputs)


class PositionEmbedding(nn.Module):
    ''' position embedding '''

    def __init__(self, n_position, embed_dim):
        def _get_sinusoid_encoding_table(n_position, embed_dim, padding_idx=None):
            ''' Sinusoid position encoding table '''

            def __cal_angle(position, hid_idx):
                return position / np.power(10000, 2 * (hid_idx // 2) / embed_dim)

            def __get_posi_angle_vec(position):
                return [__cal_angle(position, hid_j) for hid_j in range(embed_dim)]

            sinusoid_table = np.array(
                [__get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

            sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
            sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

            if padding_idx is not None:
                sinusoid_table[padding_idx] = 0

            return torch.FloatTensor(sinusoid_table)

        self.embedding = WordEmbedding(
            n_position, embed_dim, True,
            _get_sinusoid_encoding_table(n_position, embed_dim, padding_idx=0),
            False)

    def foward(self, inputs):
        return self.embedding(inputs)


class BERTEmbeddings(nn.Module):
    def __init__(self, config):
        super(BERTEmbeddings, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
        """
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BERTLayerNorm(config)  # pylint: disable=invalid-name
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
