import torch.nn as nn

from .embedding import WordEmbedding, PositionEmbedding
from .transform import RNN


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