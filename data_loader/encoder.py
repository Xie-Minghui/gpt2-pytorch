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


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char

    return pairs


class Encoder():

    def __init__(self, tokens2id, bpe_ranks):
        self.tokens2id = tokens2id
        self.bpe_ranks = bpe_ranks
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.bytes2unicode = bytes_to_unicode()
        self.unicode2bytes = {v: k for k, v in self.bytes2unicode}
        self.cache = {}

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            i = 0
            new_word = []
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.extend(word[i])
                    i += 1
            word = tuple(new_word)

            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        word = ' '.join(word)
        self.cache[token] = word

        return word

    def encode(self, context):

        bpe_tokens_id = []
        for tokens in re.findall(self.pat, context):
            tokens = ''.join([self.bytes2unicode[cha] for cha in tokens.encode('utf-8')])
            bpe_tokens_id.extend([self.tokens2id[token] for token in self.bpe(tokens).split(' ')])
        return bpe_tokens_id


def get_encoder():
    with open('../data/encoder.json', 'r') as f:
        tokens2id = json.load(f)

    with open('vocvab.bpe', 'r') as f:
        bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

    return Encoder(tokens2id, bpe_ranks)


