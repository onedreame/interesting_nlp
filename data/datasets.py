# coding=utf-8
import os
import sys
import json
import pickle
from tqdm import tqdm
from itertools import chain
import torch
from torch.nn.utils.rnn import pad_sequence

sys.path.append(os.getcwd())
from data.base import BaseDataset

__all__ = ['LSCCDataSet', 'THUCNewsDataset', 'CharLevelDataset']


class LSCCDataSet(BaseDataset):
    '''清华大学公布的一个大规模bot语料'''
    __special_tokens = ["[speaker1]", "[speaker2]"]

    @classmethod
    def get_identity_id(cls, tokenizer):
        return tokenizer.convert_tokens_to_ids(cls.__special_tokens)

    def __init__(self, path=None, tokenizer=None, max_history=15, cache_dir="cache", **kwargs):
        super(LSCCDataSet, self).__init__(tokenizer)
        self.max_history = max_history
        self.speaker1, self.speaker2 = tokenizer.convert_tokens_to_ids(LSCCDataSet.__special_tokens)
        self.data = self.load_data(path)
        cache_file = os.path.join(cache_dir, "_".join([path.replace("/", "_"), str(len(self.data))]))
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        if os.path.exists(cache_file) and os.path.isfile(cache_file):
            with open(cache_file, 'rb') as f:
                self.convs = pickle.load(f)
        else:
            self.convs = [LSCCDataSet.encode(conv[-2 * self.max_history:-1], [conv[-1]],
                                             tokenizer, self.speaker1, self.speaker2)
                          for conv in tqdm(self.data)]
            with open(cache_file, 'wb') as f:
                pickle.dump(self.convs, f)

    def load_data(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    @classmethod
    def encode(cls, history, reply, tokenizer, speaker1, speaker2, with_sep=True, do_encode=True):
        '''
        tokens转化为ids 序列，为了防止会话长度过长，通过max_history进行截断
        :param history: 历史会话序列
        :param reply: 响应
        :param tokenizer: 分词器
        :param speaker1: int, 会话者1标识
        :param speaker2: int, 会话者2标识
        :param set_up: bool, 是否需要进行初始化设置
        :return: dict
        '''
        if do_encode:
            history = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(seq)) for seq in history]
            reply = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(seq)) +
                     [tokenizer.sep_token_id if with_sep else []] for seq in reply]
        seqs = history + reply
        input_ids = [[tokenizer.cls_token_id]] + \
                    [
                        [speaker1 if i % 2 == 0 else speaker2] + seq
                        for i, seq in enumerate(seqs)
                    ]
        token_type_ids = [tokenizer.cls_token_id] + [speaker1 if i % 2 == 0 else speaker2
                                                     for i, s in enumerate(input_ids[1:])
                                                     for _ in s]
        lm_labels = [-100 for i, s in enumerate(input_ids[:-1]) for _ in s] + [-100] + input_ids[-1][1:]
        return {
            "input_ids": cls.flatten(input_ids),
            "token_type_ids": token_type_ids,
            "lm_labels": lm_labels
        }

    def __len__(self):
        return len(self.convs)

    def __getitem__(self, item):
        return self.convs[item]

    def collate(self, batch):
        if self.tokenizer._pad_token is None:
            pad_value = 0
        else:
            pad_value = self.tokenizer.pad_token_id
        input_ids = pad_sequence([torch.tensor(b['input_ids'], dtype=torch.long) for b in batch],
                                 batch_first=True, padding_value=pad_value)
        token_type_ids = pad_sequence([torch.tensor(b['token_type_ids'], dtype=torch.long) for b in batch],
                                      batch_first=True, padding_value=pad_value)
        lm_labels = pad_sequence([torch.tensor(b['lm_labels'], dtype=torch.long) for b in batch],
                                 batch_first=True, padding_value=-100)
        return input_ids, token_type_ids, lm_labels

class THUCNewsDataset(BaseDataset):
    def __init__(self, path, label_path, tokenizer, max_len=512, cache_dir="cache", **kwargs):
        super(THUCNewsDataset, self).__init__(path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        cache_file = os.path.join(cache_dir, "_".join([path.replace("/", "_"), str(len(self.data))]))
        with open(label_path, 'r') as f:
            self.label2id = json.load(f)
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        if not os.path.exists(cache_file) and os.path.isfile(cache_file):
            with open(cache_file, 'rb') as f:
                self.examples = pickle.load(f)
        else:
            self.examples = []
            for seq in tqdm(self.data):
                label_text, seq = seq.strip().split('\t')
                label_int = self.label2id[label_text]
                # seq, label_text = seq.strip().split('\t')
                # label_int = int(label_text)
                self.examples.append(self.encode(seq, tokenizer, self.max_len, label=label_int))
            with open(cache_file, 'wb') as f:
                pickle.dump(self.examples, f)

    @staticmethod
    def encode(seq, tokenizer, max_len, do_encode=True, label=-1):
        '''
        tokens转化为ids 序列，通过max_len进行截断
        :param seq: 训练句子，格式为（label\t sentence)
        :param max_len: 最长序列长度
        :param tokenizer
        :param do_encode: 是否将str序列编码为id序列
        :return: dict
        '''
        if do_encode:
            seq = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(seq))
        seq = seq[:max_len]
        attn_mask = [1] * len(seq)
        segment_ids = [1] * len(seq)
        return {
            "input_ids": seq,
            "token_type_ids": segment_ids,
            "attn_mask": attn_mask,
            "label": label
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    def collate(self, batch):
        if not hasattr(self.tokenizer, '_pad_token') or self.tokenizer._pad_token is None:
            pad_value = 0
        else:
            pad_value = self.tokenizer.pad_token_id
        input_ids = pad_sequence([torch.tensor(b['input_ids'], dtype=torch.long) for b in batch],
                                 batch_first=True, padding_value=pad_value)
        token_type_ids = pad_sequence([torch.tensor(b['token_type_ids'], dtype=torch.long) for b in batch],
                                      batch_first=True, padding_value=pad_value)
        attn_mask = pad_sequence([torch.tensor(b['attn_mask'], dtype=torch.long) for b in batch],
                                 batch_first=True, padding_value=pad_value)
        label = torch.LongTensor([b['label'] for b in batch])
        return input_ids, token_type_ids, attn_mask, label

class CharLevelDataset(BaseDataset):
    def __init__(self, path, tokenizer, seq_len=30, cache_dir="cache"):
        super(CharLevelDataset, self).__init__(tokenizer)
        self.seq_len = seq_len
        cache_file = os.path.join(cache_dir, path.replace("/", "_"))
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        if os.path.exists(cache_file) and os.path.isfile(cache_file):
            with open(cache_file, 'rb') as f:
                self.word_seqs = pickle.load(f)[:20000]
                print("小说总长度：", len(self.word_seqs))
                # self.word_seqs = self.word_seqs[:int(len(self.word_seqs)*0.3)]
        else:
            with open(path, 'r') as f:
                self.word_seqs = list(chain(*[self.encode(tokenizer, line.strip()) for i, line in enumerate(f)]))
            with open(cache_file, 'wb') as f:
                pickle.dump(self.word_seqs, f)

    @staticmethod
    def encode(tokenizer, sent):
        '''
        tokens转化为ids 序列
        :param tokenizer: 分词器
        :param sent: str, 句子序列
        :return: list
        '''
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))

    def __len__(self):
        return len(self.word_seqs) // self.seq_len

    def __getitem__(self, item):
        words = self.word_seqs[item * self.seq_len : min(len(self.word_seqs) - 1, (item + 1) * self.seq_len)] + \
               [self.tokenizer.eos_token_id]
        labels = self.word_seqs[item * self.seq_len + 1: min(len(self.word_seqs), (item + 1) * self.seq_len) + 1] + \
              [self.tokenizer.eos_token_id]
        return {"inputs": words, "labels": labels}

    def collate(self, batch):
        if self.tokenizer.pad_token is None:
            pad_value = 0
        else:
            pad_value = self.tokenizer.pad_token_id
        input_ids = pad_sequence([torch.tensor(b['inputs'], dtype=torch.long) for b in batch],
                                  batch_first=True, padding_value=pad_value)
        input_lens = torch.tensor([len(b) for b in batch], dtype=torch.long)
        labels = pad_sequence([torch.tensor(b['labels'], dtype=torch.long) for b in batch],
                               batch_first=True, padding_value=pad_value)
        return input_ids, input_lens, labels

if __name__ == '__main__':
    # LSCCDataSet('../chatbot/datasets/lscc/LCCC-base_train.json',)
    from module import BasicTokenizer
    tokenizer = BasicTokenizer('/root/PycharmProjects/interesting_nlp/datasets/doupo/label.tsv',need_tokenize=False)
    # print(tokenizer.vocab)
    # print(tokenizer.pad_token)
    # print(tokenizer.pad_token_id)
    print(tokenizer.decode([162, 1948, 2951, 3517, 3066, 163, 1075, 3692, 982, 3962, 200, 2265, 1061, 1990, 4752, 2066, 240, 212]))
    print(tokenizer.decode(torch.tensor([ 162, 1948, 2951, 3517, 3066,  163, 1075, 3692,  982, 3962,  200, 2265,
         1061, 1990, 4752, 2066,  240,  212,  204,  204, 1279, 3915, 3289,    8,
         4785,   14, 4785,   14, 4785,    9, 4786], dtype=torch.int)))
    for b in torch.tensor([ 162, 1948, 2951, 3517, 3066,  163, 1075, 3692,  982, 3962,  200, 2265,
         1061, 1990, 4752, 2066,  240,  212,  204,  204, 1279, 3915, 3289,    8,
         4785,   14, 4785,   14, 4785,    9, 4786]).tolist():
        print(b)

    val_data = CharLevelDataset('/root/PycharmProjects/interesting_nlp/datasets/doupo/train.txt', tokenizer)
    iter = torch.utils.data.DataLoader(val_data, batch_size=4, num_workers=2,collate_fn=val_data.collate)
    print(val_data.word_seqs)
    print("$$$$$$$$$$$$$$$$$$\n", tokenizer.decode(val_data.word_seqs))
    for batch in iter:
        inputs, lens = tuple(b for b in batch)
        print(inputs)
        print(inputs.dtype)
        print(lens)
        for b in inputs:
            print("raw: ", b)
            print(tokenizer.decode(b.tolist()))
