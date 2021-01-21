# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/1/21 17:37
# @File    : encoder.py

"""
file description:：

"""
import json
import regex as re
from data_loader.global_process import bytes_to_unicode

class Encoder():

    def __init__(self, tokens2id, bpe_rank):
        self.tokens2id = tokens2id
        self.bpe_rank = bpe_rank
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.bytes2unicode = bytes_to_unicode()
        self.unicode2bytes = {v:k for k, v in self.bytes2unicode}

    def bpe(self, token):

        pass

    def encode(self, context):

        bpe_tokens_id = []
        for tokens in re.findall(self.pat, context):
            tokens =  ''.join([self.bytes2unicode[cha] for cha in tokens.encode('utf-8')])
            bpe_tokens_id.extend([self.tokens2id[token] for token in self.bpe(tokens)])
        return bpe_tokens_id




with open('../data/encoder.json', 'r') as f:
    tokens2id = json.load(f)

with open('vocvab.bpe', 'r') as f:
    bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))


