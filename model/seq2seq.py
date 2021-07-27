import random

import torch
import torch.nn as nn

from module import EncoderRNN, DecoderRNN

__all__ = ['Seq2Seq']


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers, dropout,
                 max_len, attn_model='concat', sos=0, eos=0, **kwargs):
        super(Seq2Seq, self).__init__()
        self.use_converage = kwargs.get("use_converage", False)
        encoder = EncoderRNN(vocab_size, hidden_size, n_layers, dropout)
        decoder = DecoderRNN(attn_model, hidden_size, vocab_size, n_layers, dropout, use_coverage=self.use_converage)
        if kwargs.get('share_emb', False):
            decoder.embedding = encoder.embedding

        self.encoder = encoder
        self.decoder = decoder
        self.sos = sos
        self.eos = eos
        self.max_len = max_len

        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def get_input_mask(self, src_lengths):
        """
        生成输入序列的mask，用于覆盖机制
        : param src_lengths: (batch_size, )
        """
        max_len = src_lengths.max().item()
        mask = torch.arange(max_len, dtype=torch.float).repeat(src_lengths.size(0), 1).type_as(src_lengths)
        src_lengths_2d = src_lengths.view(-1, 1)
        mask[mask < src_lengths_2d] = 1
        mask[mask >= src_lengths_2d] = 0
        return mask

    def forward(self, src, tgt=None, src_lengths=None, teacher_forcing_ratio=0.2):
        if self.use_converage:
            coverage_tensor = torch.zeros_like(src.size(), dtype=torch.float).type_as(src)
            coverage_mask = self.get_input_mask(src_lengths)
        else:
            coverage_mask = None
            coverage_tensor = None
        encoder_outputs, encoder_hidden = self.encoder(src, src_lengths, None)

        # Prepare input and output variables
        this_batch_size = src.size()[0]
        decoder_input = torch.LongTensor([self.sos] * this_batch_size).type_as(src)
        # 对于多层的gru，要排除掉后向的hidden，只使用前向的hidden
        decoder_hidden = encoder_hidden.view(self.encoder.n_layers, -1,
                                             *encoder_hidden.size()[-2:])[:, 0, ...].contiguous()

        max_target_length = tgt.size()[1] if tgt is not None else self.max_len
        decoder_outputs = []

        # Run through decoder one time step at a time
        for t in range(max_target_length):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, coverage_tensor, coverage_mask
            )

            use_teacher_forcing = random.random() < teacher_forcing_ratio
            decoder_outputs.append(decoder_output.unsqueeze(1))
            decoder_input = tgt[:, t] if use_teacher_forcing else decoder_output.argmax(-1)
        return torch.cat(decoder_outputs, dim=1)

    def predict(self, src, src_lengths):
        encoder_outputs, encoder_hidden = self.encoder(src, src_lengths, None)

        # Prepare input and output variables
        this_batch_size = src.size()[0]
        decoder_input = torch.LongTensor([self.sos] * this_batch_size).type_as(src)
        # 对于多层的gru，要排除掉后向的hidden，只使用前向的hidden
        decoder_hidden = encoder_hidden.view(self.encoder.n_layers, -1,
                                             *encoder_hidden.size()[-2:])[:, 0, ...].contiguous()

        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

        return decoder_output
