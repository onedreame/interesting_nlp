#!/usr/bin/env python
# coding:utf8

# Copyright (c) 2019, Tencent. All rights reserved
# Author: Tang Jing (jamesjtang@tencent.com)

# Provide data utilies for each task's data processing.


import os
import json
import torch.utils.data as data

__all__ = ['SingleFIleDataset', 'MultiFilesDataset']


def truncate_seq_pair(max_seq_len, tokens_a, tokens_b):
    """Truncate the sequence of pair, the last token will be removed
    if it is longer than the other.
    Args:
        max_seq_len: max length of target sequence
        tokens_a: first token sequence of the pair
        tokens_b: second token sequence of the pair
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_seq_len - 2:  # for [CLS] and [SEP] tokens
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def find_first_sublist(main_list, sub_list):
    """Find the start and end indexes of sublist in main list
    Args:
        main_list: the main list.
        sub_list: the sublist.
    Return:
        the start and end indexes of sub_list in main_list
    """
    sub_len = len(sub_list)
    for i, _ in enumerate(main_list):
        if main_list[i: i + sub_len] == sub_list:
            return (i, i + sub_len - 1)


class BaseDataset(data.Dataset):
    '''base class of different task dataset'''

    def __init__(self, tokenizer=None, **kwargs):
        self.tokenizer = tokenizer

    def encode(self, sent):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sent))

    def decode(self, tokens, *args, **kwargs):
        return self.tokenizer.decode(tokens)

    def load_data(self, path:str):
        raise NotImplementedError

    @classmethod
    def build_dataset(cls, config:dict):
        return cls(**config)

    @property
    def name(self):
        return self.__class__.__name__

class MultiFilesDataset(BaseDataset):
    '''
        The multi files class of different task dataset
        only support text files
    '''

    def __init__(self, file_path: str = None):
        super(MultiFilesDataset, self).__init__(file_path)
        self.file_path = file_path
        self.files_list = list()
        self.files_offset = list()
        self._load_data()

    def _read_txt(self):
        """
            Fill the file index and offsets of each line in files_list in offset_list
            Args:
                path: string of file path, support single file or file dir
                files_list: the list contains file names
                offset_list: the list contains the tuple of file name index and offset
        """
        if os.path.isdir(self.file_path):  # for multi-file, its input is a dir
            self.files_list.extend([os.path.join(self.file_path, f) for f in os.listdir(self.file_path)])
        elif os.path.isfile(self.file_path):  # for single file, its input is a file
            self.files_list.append(self.file_path)
        else:
            raise RuntimeError(self.file_path + " is not a normal file.")
        for i, file_path in enumerate(self.files_list):
            offset = 0
            with open(file_path, "r", encoding="utf-8") as single_file:
                for line in single_file:
                    tup = (i, offset)
                    self.files_offset.append(tup)
                    offset += len(bytes(line, encoding='utf-8'))

    def __len__(self):
        return len(self.files_offset)

    def _get_line(self, index):
        tup = self.files_offset[index]
        target_file = self.files_list[tup[0]]
        with open(target_file, "r", encoding="utf-8") as input_file:
            input_file.seek(tup[1])
            line = input_file.readline()
        return line


if __name__ == '__main__':
    class A(object):
        def __init__(self):
            self.a = 1

        @property
        def name(self):
            return self.__class__.__name__


    class B(A):
        pass


    a = A()
    print(a.name)
    b = B()
    print(b.name)
