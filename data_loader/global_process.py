# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/1/21 17:06
# @File    : global_process.py
"""
file description：
to handle global data
"""
from functools import lru_cache


@lru_cache()
def bytes_to_unicode():
    '''
    to map tokens not in bpe file to special token in bpe file.
    :return: a map of id and str
    :rtype: dict
    '''
    id_list = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    str_list = id_list[:]

    cnt = 0
    for i in range(2**8):
        if i not in id_list:
            id_list.append(i)
            str_list.append(2**8+cnt)
            cnt += 1
    str_list = [chr(i) for i in str_list]

    return dict(zip(id_list, str_list))


