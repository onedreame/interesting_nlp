import torch
import torch.nn as nn

from .embedding import PositionEmbedding, WordEmbedding
from .transform import RNN
from .attention import Attention


class RNNDecoder(nn.Module):
    def __init__(self, vocab, conf):
        super(RNNDecoder, self).__init__()

        _, tgt_vocab_size = vocab.vocab_size()
        embed_dim = conf.data.embed_dim
        use_pretrained_embed = conf.data.use_pretrained_embed
        _, tgt_pretrained_embed = vocab.embedding()
        fine_tune = conf.data.fine_tune

        self.word_emb = WordEmbedding(
            tgt_vocab_size, embed_dim, use_pretrained_embed,
            tgt_pretrained_embed, fine_tune)

        self.use_position_emb = conf.data.use_position_emb
        if self.use_position_emb:
            n_position = conf.data.max_seq_len
            self.position_emb = PositionEmbedding(n_position, embed_dim)

        model_name = conf.decoder.model_name
        self.m_conf = None

        self.decoder, self.attention, self.trans = None, None, None

        if model_name == "RNN":
            self.m_conf = conf.decoder.RNN
            hidden_size = self.m_conf.hidden_size
            self.decoder = RNN(self.m_conf.rnn_type,
                               self.m_conf.input_size,
                               hidden_size,
                               self.m_conf.num_layers,
                               self.m_conf.dropout,
                               self.m_conf.bidirectional)
            if self.m_conf.use_attention:
                self.attention = Attention(hidden_size)
                self.trans = nn.Linear(hidden_size * 2, hidden_size)
        else:
            raise RuntimeError("Unknown Model: " + model_name)

    def forward(self, tgt_seq, tgt_len, src_enc, src_hiddens, tgt_pos=None):
        tgt_emb = self.word_emb(tgt_seq)
        if self.use_position_emb:
            tgt_emb += self.position_emb(tgt_pos)

        output, hiddens = self.decoder(tgt_emb, tgt_len, hiddens=src_hiddens)
        output = output.squeeze(1)

        if self.m_conf.use_attention:
            context = self.attention(output, src_enc, src_enc)
            output = self.trans(torch.cat((output, context), 1))

        return output, hiddens