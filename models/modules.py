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
    def __init__(self, d_model, hidden_dim, dropout):
        super().__init__()
        self.c_fc = Conv1D(d_model, hidden_dim)
        self.c_project = Conv1D(hidden_dim, d_model)
        self.activation = gelu()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout_layer(self.c_project(self.activation(self.c_fc(x))))








