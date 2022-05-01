import torch.nn as nn

from .embedding import WordEmbedding, PositionEmbedding
from .transform import RNN

__all__ = ['EncoderRNN']

class EncoderRNN(nn.Module):
    '''
    用于seq2seq结构的encoder结构，也可以用于hierarchy seq2seq的utterance encoder
    '''

    def __init__(self, vocab_size, hidden_size, n_layers=1, dropout=0.1, bidirectional=True, **kwargs):
        super(EncoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.num_direction = 2 if bidirectional else 1
        self.dropout = dropout

        # self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.embedding = WordEmbedding(vocab_size, hidden_size, **kwargs)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          bidirectional=bidirectional, dropout=self.dropout)

    def forward(self, input_seqs, input_lengths, hidden=None):
        '''
        使用hierarchy seq2seq的时候，batch_first应该设为true
        :param input_seqs:
            输入序列，shape为[bs, maxlen]
        :param input_lengths: 输入序列长度，shape:（bs,）
        :param hidden:   [num_layers * bidirectional, bs, max_len]
        :return:（outputs, encoder_hidden)
        '''
        # hred模型因为encoderRNN涉及到对话顺序的概念，所以手动sort utterance长度
        input_lengths_sorted, indices = input_lengths.sort(descending=True)
        input_seqs_sorted = input_seqs.index_select(0, indices)

        embedded = self.embedding(input_seqs_sorted)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths_sorted.cpu(),
                                                   batch_first=True)
        self.gru.flatten_parameters()
        outputs, hidden = self.gru(packed, hidden)
        # unpack (back to padded), outputs:
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True
                                                      )
        _, indices_reverse = indices.sort()
        outputs = outputs.index_select(0, indices_reverse)
        hidden = hidden.index_select(1, indices_reverse)
        # Sum bidirectional outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden


class ContextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(ContextRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.gru = nn.GRU(input_size, hidden_size, n_layers,
                          bidirectional=True, dropout=self.dropout)

    def forward(self, input_seqs, input_lengths, hidden=None, batch_first=True):
        '''context RNN，直接使用encoderRNN的输出作为输入，不需要额外的embedding'''
        input_lengths_sorted, indices = input_lengths.sort(descending=True)
        input_seqs_sorted = input_seqs.index_select(0, indices)
        packed = nn.utils.rnn.pack_padded_sequence(input_seqs_sorted, input_lengths_sorted.cpu(),
                                                   batch_first=batch_first)
        self.gru.flatten_parameters()
        outputs, hidden = self.gru(packed, hidden)
        # unpack (back to padded), outputs:
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=batch_first)
        _, indices_reverse = indices.sort()
        outputs = outputs.index_select(0, indices_reverse)
        hidden = hidden.index_select(1, indices_reverse)
        # Sum bidirectional outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden


class RNNEncoder(nn.Module):
    def __init__(self, vocab, conf):
        super(RNNEncoder, self).__init__()
        src_vocab_size, _ = vocab.vocab_size()
        embed_dim = conf.data.embed_dim
        use_pretrained_embed = conf.data.use_pretrained_embed
        src_pretrained_embed, _ = vocab.embedding()
        fine_tune = conf.data.fine_tune

        self.word_emb = WordEmbedding(
            src_vocab_size, embed_dim, use_pretrained_embed,
            src_pretrained_embed, fine_tune)

        self.use_position_emb = conf.data.use_position_emb
        if self.use_position_emb:
            n_position = conf.data.max_seq_len
            self.position_emb = PositionEmbedding(n_position, embed_dim)

        model_name = conf.encoder.model_name
        if model_name == "RNN":
            m_conf = conf.encoder.RNN
            self.encoder = RNN(m_conf.rnn_type,
                               m_conf.input_size,
                               m_conf.hidden_size,
                               m_conf.num_layers,
                               m_conf.dropout,
                               m_conf.bidirectional)
        else:
            raise RuntimeError("Unknown Model: " + model_name)

    def forward(self, src_seq, src_len, src_pos=None):
        src_emb = self.word_emb(src_seq)
        if self.use_position_emb:
            src_emb += self.position_emb(src_pos)

        output, hiddens = self.encoder(src_emb, src_len)

        return output, hiddens