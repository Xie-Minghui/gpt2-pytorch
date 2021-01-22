# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/1/22 9:01
# @File    : modules.py

"""
file description:：

"""
import torch.nn as nn
import torch
import math


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Modules):

    def __init__(self, normalized_shape, epsilon=1e-12):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.epsilon = epsilon

    def forward(self, x):
        x_expected = x.mean(-1, keepdims=True)
        x_var = (x - x_expected).pow(2).mean(-1, keepdims=True)
        x = (x - x_expected) / torch.sqrt(x_var+self.epsilon)
        return x * self.weights + self.bias


class Conv1D(nn.Modules):
    '''
    The CONV1D layer can be thought of as a LINEAR layer itself. Essentially, it is casting an initial tensor x
     (having the final dimension of x.size(-1)) being passed to it to have a final dimension of size self.output_dim
    '''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = torch.empty(input_dim, output_dim)
        nn.init.normal_(self.weights, std=0.02)
        self.weights = nn.Parameter(self.weights)
        self.bias = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.output_dim,)
        x = torch.addmm(self.bias, x.view(-1, self.input_dim), self.weights)
        x = x.view(*size_out)
        return x


class FeedForwardNetwork(nn.Modules):
    def __init__(self, n_embed, hidden_dim, dropout):
        super().__init__()
        self.c_fc = Conv1D(n_embed, hidden_dim)
        self.c_project = Conv1D(hidden_dim, n_embed)
        self.activation = gelu()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout_layer(self.c_project(self.activation(self.c_fc(x))))


class Attention(nn.Modules):

    def __init__(self, n_embed=768, n_ctx=1024, scale=True):
        super().__init__()
        self.n_embed = n_embed
        self.scale = scale
        self.atten_head = 12

        self.regester_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))

        self.c_atten = Conv1D(n_embed, n_embed*3)
        self.c_project = Conv1D(n_embed, n_embed)

        assert self.n_embed % self.atten_head == 0, "n_embed must mod atten_head"

    # def split_heads(self, x, is_key=False):
    #     x_new_shape = x.size()[:-1] + (self.atten_head, self.n_embed//self.atten_head)
    #     x = x.view(*x_new_shape)
    #     if is_key:
    #         return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
    #     else:
    #         return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def split_heads(self, x):
        x_new_shape = x.size()[:-1] + (self.atten_head, self.n_embed//self.atten_head)
        x = x.view(*x_new_shape)

        return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        x_new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1))

        return x.view(*x_new_shape)

    def mask_attn_weights(self, atten_weight):
        start, end = atten_weight.size(-2), atten_weight.size(-1)
        mask = self.bias[:, :, end-start:end, :end]  # mask的意义不是很清楚
        atten_weight = atten_weight * mask - 1e10 * (1-b)

        return atten_weight

    def cal_attention(self, query, key, value):
        atten_weight = torch.matmul(query, key.transpose(-2,-1))
        if self.scale:
            # atten_weight [batch, head, seq_length, seq_length]
            atten_weight = atten_weight / math.sqrt(value.szie(-1))  # head_features
        atten_weight = self.mask_attn_weights(atten_weight)
        atten_weight = nn.Softmax(dim=-1)(atten_weight)

        return torch.matmul(atten_weight, value)

    def forward(self, x, layer_past=None):
        x = self.c_atten(x)
        query, key, value = x.split(self.n_embed, dim=2)
        query, key, value = map(self.split_heads, (query, key, value))
        # query = self.split_heads(query)
        # key = self.split_heads(key)
        # value = self.split_heads(value)

        if layer_past:
            key_past, value_past = layer_past[0].transpose(-2, -1), layer_past[1]
            key = torch.cat((key, key_past), dim=-1)
            value = torch.cat((value, value_past), dim=-1)
        # Concatenate a sequence of tensors along a new dimension

        out = self.cal_attention(query, key, value)
        out = self.merge_heads(out)
        out = self.c_project(out)

        return out


class TransformerDecoderBlock(nn.Modules):
    def __init__(self, config, scale=True):
        super().__init__()
        self.layer_norm1 = LayerNorm(config.n_embed, config.layer_norm_epsilon)
        self.atten_layer = Attention(config.n_embed, config.n_ctx, config, scale)
        self.ffn = FeedForwardNetwork(config.n_embed, 4*config.n_embed)
        self.layer_norm2 = LayerNorm(config.n_embed, config.layer_norm_epsilon)

    def forward(self, x, layer_past):
        x = x + self.atten_layer(self.layer_norm1(x), layer_past=layer_past)
        x = x + self.ffn(self.layer_norm2(x))

        return x








