import torch
import torch.nn as nn

from .embedding import PositionEmbedding, WordEmbedding
from .transform import RNN
from .attention import Attention

__all__ = ['DecoderRNN']

class DecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1, use_coverage=False,
                 use_pretrained_embed=False, pretrained_embed=None, fine_tune=True, **kwargs):
        super(DecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_coverage = use_coverage

        # Define layers
        # self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding = WordEmbedding(
            output_size, hidden_size, use_pretrained_embed,
            pretrained_embed, fine_tune)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.embedding_dropout = nn.Dropout(dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.generator = nn.Linear(hidden_size, output_size)
        self.attn_model = attn_model

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attention(hidden_size, attn_model, use_coverage)

    def forward(self, input_seq, context, encoder_outputs=None, coverage_tensor=None, coverage_mask=None):
        # Get the embedding of the current input word (last output word)
        input_seq = input_seq.unsqueeze(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        self.gru.flatten_parameters()
        rnn_output, hidden = self.gru(embedded, context)

        if self.attn_model != "none":
            # Calculate attention from current RNN state and all encoder outputs;
            # apply to encoder outputs to get weighted average
            # 这里使用最底层的hidden
            context = self.attn(hidden[0], encoder_outputs, encoder_outputs, coverage_tensor, coverage_mask)  # B 1 H
            # context = attn_weights.bmm(encoder_outputs)  # B 1 H

            # Attentional vector using the RNN hidden state and context vector
            # concatenated together (Luong eq. 5)
            # context = context.transpose(0, 1)  # 1 x B x N
            concat_input = torch.cat((rnn_output.squeeze(0), context.squeeze(1)), -1)  # B × 2*N
            concat_output = torch.tanh(self.concat(concat_input))

            # Finally predict next token (Luong eq. 6, without softmax)
            output = self.generator(concat_output)
        else:
            output = self.generator(rnn_output.squeeze(0))
        return output, hidden

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