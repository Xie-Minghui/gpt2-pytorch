# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/1/24 20:21
# @File    : main.py

"""
file description:：

"""

from data_loader.encoder import get_encoder
from models.modules import GPT2GeneratorModel
from utils.config import GPT2Config
from utils.transfer import load_weight

import argparse
import os
import torch
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text', required=False, default='i am a boy,')
    parser.add_argument('-b', '--bsz', type=int, default=1)
    parser.add_argument('-gl', '--generate_length', type=int, default=-1)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')

    args = parser.parse_args()

    return args


def text_generator(state_dict):
    config = GPT2Config()
    args = parse_args()
    context = args.text
    if args.generate_length == -1:
        args.generate_length == config.n_ctx // 4

    enc = get_encoder()

    context_ids = enc.encode(context)

    gpt2_model = GPT2GeneratorModel(config)

    load_weight(gpt2_model, state_dict)

    output = gpt2_model.generate_sequence(context_ids,
                                          start_token=enc.tokens2id['<|endoftext|>'] if args.unconditional else None,
                                          generate_length=args.generate_length
                                          )

    output = output[:,len(context_ids):]

    print('='*40 + ' SAMPLE ' + '='*40)
    print(output)


if __name__ == '__main__':
    if os.path.exists('../gpt2-pytorch_model.bin'):
        state_dict = torch.load('../gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
        text_generator(state_dict)
    else:
        print('Please download gpt2-pytorch_model.bin')
        sys.exit()








