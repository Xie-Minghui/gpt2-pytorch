# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/1/22 9:01
# @File    : Module.py

"""
file description:：

"""
import torch.nn as nn
import torch
import math
import copy
import torch.nn.functional as F
from tqdm import trange


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, epsilon=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.epsilon = epsilon

    def forward(self, x):
        x_expected = x.mean(-1, keepdims=True)
        x_var = (x - x_expected).pow(2).mean(-1, keepdims=True)
        x = (x - x_expected) / torch.sqrt(x_var+self.epsilon)
        return self.weight * x + self.bias


class Conv1D(nn.Module):
    '''
    The CONV1D layer can be thought of as a LINEAR layer itself. Essentially, it is casting an initial tensor x
     (having the final dimension of x.size(-1)) being passed to it to have a final dimension of size self.output_dim
    '''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = torch.empty(input_dim, output_dim)
        nn.init.normal_(self.weight, std=0.02)
        self.weight = nn.Parameter(self.weight)
        self.bias = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.output_dim,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)  # self.input_dim
        x = x.view(*size_out)
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, n_embed, hidden_dim, dropout=0.1):
        super().__init__()
        self.c_fc = Conv1D(n_embed, hidden_dim)
        self.c_project = Conv1D(hidden_dim, n_embed)
        self.activation = gelu
        self.dropout_layer = nn.Dropout(dropout)  # 测试的时候需要去掉

    def forward(self, x):
        return self.c_project(self.activation(self.c_fc(x)))


class Attention(nn.Module):

    def __init__(self, n_embed=768, n_ctx=1024, n_head=6, scale=False):
        super().__init__()
        self.n_embed = n_embed
        self.scale = scale
        self.atten_head = n_head

        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))  # mask

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
        x_new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)

        return x.view(*x_new_shape)

    def mask_attn_weight(self, atten_weight):
        start, end = atten_weight.size(-2), atten_weight.size(-1)
        mask = self.bias[:, :, end-start:end, :end]  # mask的意义不是很清楚
        atten_weight = atten_weight * mask - 1e10 * (1-mask)

        return atten_weight

    def cal_attention(self, query, key, value):
        atten_weight = torch.matmul(query, key.transpose(-2, -1))
        if self.scale:
            # atten_weight [batch, head]
            atten_weight = atten_weight / math.sqrt(value.size(-1))  # head_features
        atten_weight = self.mask_attn_weight(atten_weight)
        atten_weight = nn.Softmax(dim=-1)(atten_weight)

        return torch.matmul(atten_weight, value)  # 计算注意力向量

    def forward(self, x, layer_past=None):
        x = self.c_atten(x)
        query, key, value = x.split(self.n_embed, dim=2)
        query, key, value = map(self.split_heads, (query, key, value))
        # query = self.split_heads(query)
        # key = self.split_heads(key)
        # value = self.split_heads(value)

        if layer_past is not None:  # tensor无法判断真假
            key_past, value_past = layer_past[0], layer_past[1]  # 为什么要添加过去的k,v值,而q不用
            key = torch.cat((key_past, key), dim=-2)
            value = torch.cat((value_past, value), dim=-2)
        # Concatenate a sequence of tensors along a new dimension
        present = torch.stack((key, value))
        out = self.cal_attention(query, key, value)
        out = self.merge_heads(out)
        out = self.c_project(out)

        return out, present


class TransformerDecoderBlock(nn.Module):
    def __init__(self, config, scale=True):
        super().__init__()
        self.layer_norm1 = LayerNorm(config.n_embed, config.layer_norm_epsilon)
        self.atten_layer = Attention(config.n_embed, config.n_ctx, config.n_head, scale)
        self.ffn = FeedForwardNetwork(config.n_embed, 4*config.n_embed, config.dropout_rate)
        self.layer_norm2 = LayerNorm(config.n_embed, config.layer_norm_epsilon)

    def forward(self, x, layer_past=None):
        a, present = self.atten_layer(self.layer_norm1(x), layer_past=layer_past)
        x = x + a
        x = x + self.ffn(self.layer_norm2(x))

        return x, present


class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layer = config.n_layer
        self.n_embed = config.n_embed

        self.wte = nn.Embedding(config.vocab_size, config.n_embed)  # 文本嵌入向量
        self.wpe = nn.Embedding(config.n_position, config.n_embed)  # 位置编码
        block = TransformerDecoderBlock(config, scale=True)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.layer_norm = LayerNorm(config.n_embed, config.layer_norm_epsilon)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None):

        if past is None:
            past_length = 0
            past = [None] * self.n_layer
        else:
            past_length = past[0][0].size(-2)  # 当前生成文本的长度（包括初始文本）

        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        input_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0

        hidden_ids = input_embeds + position_embeds + token_type_embeds

        presents = []
        for block, layer_past in zip(self.blocks, past):
            hidden_ids, present = block(hidden_ids, layer_past)  # hidden_ids要跟着改变，翻了与GLMP一样的错误
            presents.append(present)

        hidden_ids = self.layer_norm(hidden_ids)  # [1, seq_length, n_embed]

        return hidden_ids, presents


class GPT2LMHeadModel(nn.Module):
    def __init__(self, embed_weight):
        super().__init__()
        self.share_embed_weight(embed_weight)

    def share_embed_weight(self, embed_weight):
        embed_shape = embed_weight.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = embed_weight

    def forward(self, hidden_states):
        return self.decoder(hidden_states)


class GPT2GeneratorModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gpt2 = GPT2Model(config)
        self.decoder = GPT2LMHeadModel(self.gpt2.wte.weight)

    def top_k_logits(self, logits, k):
        if k == 0:
            return logits
        values, _ = torch.topk(logits, k)
        min_values = values[:, -1].unsqueeze(1)
        return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

    def generate_sequence(self, context=None, start_token=None, generate_length=256, topk_num=40, sample=True):
        self.train(False)
        if start_token is None:
            assert context is not None, "Specify exactly one of start_token and context!"
            context = torch.tensor(context, dtype=torch.long).unsqueeze(0)
        else:
            assert context is None, "Specify exactly one of start_token and"
            context = torch.full((1,), start_token, dtype=torch.long)

        prev, output = context, context
        past = None
        with torch.no_grad():
            for _ in trange(generate_length):
                logits, past = self.forward(prev, past=past)
                logits = logits[:, -1, :]
                logits = self.top_k_logits(logits, topk_num)
                # logits ,_ = torch.topk(logits, topk_num)  # 这个会输出一对乱码
                log_soft = F.softmax(logits, dim=-1)
                if sample:
                    prev = torch.multinomial(log_soft, num_samples=1)
                else:
                    _, prev = torch.topk(log_soft, k=1, dim=-1)
                output = torch.cat((output, prev), dim=1)

        return output

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.decoder.share_embed_weight(self.gpt2.wte.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        hidden_states, presents = self.gpt2(input_ids, position_ids, token_type_ids, past)
        lm_logits = self.decoder(hidden_states)

        if lm_labels:
            loss_func = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_func(lm_logits.view(-1, lm_logits.size(-1)), lm_labels)
            return loss

        return lm_logits, presents










