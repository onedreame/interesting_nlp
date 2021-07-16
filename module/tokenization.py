# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata
import six

__all__ = ['FullTokenizer', 'BasicTokenizer']


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file, sep='\t'):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 4  # 0-3 for special token
    with open(vocab_file, "r", encoding='utf-8') as f:
        for line in f:
            tokens = convert_to_unicode(line).strip().split(sep)
            if len(tokens) > 1:
                try:
                    vocab[tokens[0]] = int(tokens[1])
                except:
                    print(f'Error line: {line}')
            else:
                vocab[tokens] = index
                index += 1
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BaseTokenizer(object):
    '''Base Tokenizer'''

    def __init__(self, vocab_file=None, need_tokenize=True, **kwargs):
        if vocab_file is None:
            raise RuntimeError("vocab_file is None!")
        self.vocab = collections.OrderedDict()
        self.build_vocab(vocab_file, need_tokenize)
        self.add_default_special(**kwargs)
        self.id2token = {id: token for token, id in self.vocab.items()}

    def add_default_special(self, **kwargs):
        """
        为tokenizer添加几个特殊的token
        """
        self.add_special("unk_token", kwargs.get("unk_token", "<unk>"))
        self.add_special("pad_token", kwargs.get("pad_token", "<pad>"))
        self.add_special("eos_token", kwargs.get("eos_token", "<eos>"))
        self.add_special("sos_token", kwargs.get("sos_token", "<sos>"))

    def add_special(self, name, token):
        if self.vocab is None:
            raise RuntimeError("Please use build_vocab to initialize vocab")
        if name not in self.vocab:
            self.vocab[token] = len(self.vocab)
        self.__setattr__(name, token)
        self.__setattr__(name + "_id", self.vocab[token])

    @property
    def vocab_size(self):
        return 0 if self.vocab is None else len(self.vocab)

    def tokenize(self, text):
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab.get(token, self.unk_token_id))
        return ids

    def convert_ids_to_tokens(self, ids):
        return [self.id2token.get(id, self.unk_token) for id in ids]

    def decode(self, ids):
        return "".join(self.convert_ids_to_tokens(ids))

    def get_token_counter(self, content_path):
        counter = collections.Counter()
        with open(content_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = self.tokenize(line.strip())
                counter.update(tokens)
        return counter

    def build_vocab(self, file_for_vocab, need_tokenize=False, min_freq=1, **kwargs):
        """
        build vocabulary
        need_tokenize: bool, if true, vocab file must be a vocabulary file
        """
        if not need_tokenize:
            self.vocab.update(load_vocab(file_for_vocab))
            return
        counter = self.get_token_counter(file_for_vocab)
        self.vocab.update({token: idx + 4 for idx, (token, freq) in enumerate(counter.items()) if freq >= min_freq})


class FullTokenizer(BaseTokenizer):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True, need_tokenize=True, **kwargs):
        self.do_lower_case = do_lower_case
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        super(FullTokenizer, self).__init__(vocab_file, need_tokenize, **kwargs)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens


class BasicTokenizer(BaseTokenizer):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, vocab_file=None, do_lower_case=True, need_tokenize=True, **kwargs):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
          vocab: Vocab.
        """
        self.do_lower_case = do_lower_case
        super(BasicTokenizer, self).__init__(vocab_file, need_tokenize, **kwargs)

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            char_code = ord(char)
            if self._is_chinese_char(char_code):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, char_code):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((char_code >= 0x4E00 and char_code <= 0x9FFF) or  #
                (char_code >= 0x3400 and char_code <= 0x4DBF) or  #
                (char_code >= 0x20000 and char_code <= 0x2A6DF) or  #
                (char_code >= 0x2A700 and char_code <= 0x2B73F) or  #
                (char_code >= 0x2B740 and char_code <= 0x2B81F) or  #
                (char_code >= 0x2B820 and char_code <= 0x2CEAF) or
                (char_code >= 0xF900 and char_code <= 0xFAFF) or  #
                (char_code >= 0x2F800 and char_code <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            char_code = ord(char)
            if char_code == 0 or char_code == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(BaseTokenizer):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab_file, need_tokenize=False, max_input_chars_per_word=100, **kwargs):
        self.max_input_chars_per_word = max_input_chars_per_word
        super(WordpieceTokenizer, self).__init__(vocab_file, need_tokenize=need_tokenize, **kwargs)

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.

        Returns:
          A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    char_code = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((char_code >= 33 and char_code <= 47) or (char_code >= 58 and char_code <= 64) or
            (char_code >= 91 and char_code <= 96) or (char_code >= 123 and char_code <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
