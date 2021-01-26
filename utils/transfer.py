# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/1/25 14:01
# @File    : transfer.py

"""
file description:：

"""
'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''

import logging

logger = logging.getLogger(__name__)


def load_weight(model, state_dict):
    old_keys = []
    new_keys = []
    cnt = 0
    for key in state_dict.keys():

        if 'h.' in key:
            new_key = key.replace('h', 'blocks',1)
            new_key = new_key.replace('ln_', 'layer_norm',1)
            new_key = new_key.replace('attn', 'atten_layer', 1)
            new_key = new_key.replace('c_attn', 'c_atten', 1)
            new_key = new_key.replace('c_proj', 'c_project', 1)
            new_key = new_key.replace('mlp', 'ffn')
            new_keys.append(new_key)
            old_keys.append(key)

        if 'ln_f' in key:
            new_key = key.replace('ln_f', 'layer_norm')
            new_keys.append(new_key)
            old_keys.append(key)

    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    pretrained_dict = {}
    for k, v in state_dict.items():
        k = 'gpt2.' + k
        if k in model_dict:
            cnt += 1
            pretrained_dict[k] = v
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.set_tied()
    model.eval()
