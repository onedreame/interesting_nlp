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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, method="dot"):
        super(Attention, self).__init__()

        self.method = method
        self.dim = dim

        if method == "dot":
            pass
        elif method == "general":
            self.line_q = nn.Linear(dim, dim, bias=False)
        elif method == "concat":
            self.line = nn.Linear(dim*2, dim, bias=False)
            self.v = nn.Parameter(torch.FloatTensor(1, dim))
        elif method == "bahdanau":
            self.line_q = nn.Linear(dim, dim, bias=False)
            self.line_k = nn.Linear(dim, dim, bias=False)
            self.attn = nn.Parameter(torch.FloatTensor(dim, dim))
        else:
            raise NotImplementedError

    def forward(self, q, k, v):
        def _score(q, k, method):
            """
            Computes an attention score
            : param q: size (batch_size, dim)
            : param k: size (batch_size, lens, dim)
            : param method: str ("dot", "general", "concat", "bahdanau")
            : return: Attention score: size (batch_size, lens)
            """
            assert q.size(-1) == self.dim
            assert k.size(-1) == self.dim

            if method == "dot":
                return k.bmm(q.unsqueeze(-1)).squeeze(-1)
            elif method == "general":
                return k.bmm(self.line_q(q).unsqueeze(-1)).squeeze(-1)
            elif method == "concat":
                out = F.tanh(self.line(torch.cat((q.unsqueeze(1).repeat(1, k.size(1), 1), k), -1))).transpose(1,2)
                return self.v.matmul(out).squeeze(1)
            elif method == "bahdanau":
                q = q.unsqueeze(1)
                out = F.tanh(self.line_q(q) + self.line_k(k))
                return out.bmm(self.attn.unsqueeze(2)).squeeze(-1)
            else:
                raise NotImplementedError

        attn_weights = _score(q, k, self.method)
        attn_weights = F.softmax(attn_weights, -1)
        normalization_factor = attn_weights.sum(1, keepdim=True)
        attn_weights = attn_weights / normalization_factor
        return attn_weights.unsqueeze(1).bmm(v).squeeze(1)

class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, inputs):
        new_x_shape = inputs.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        inputs = inputs.view(*new_x_shape)
        return inputs.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer