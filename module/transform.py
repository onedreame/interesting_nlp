import torch
import torch.nn as nn


class BERTLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """
           Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, inputs):
        mean = inputs.mean(-1, keepdim=True)
        square_mean = (inputs - mean).pow(2).mean(-1, keepdim=True)
        inputs = (inputs - mean) / torch.sqrt(square_mean + self.variance_epsilon)
        return self.gamma * inputs + self.beta

class RNN(nn.Module):
    ''' rnn module '''

    def __init__(self, rnn_type, input_size, hidden_size, num_layers, dropout,
                 bidirectional):
        super(RNN, self).__init__()

        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_first = True
        self.rnn = None

        if rnn_type == "RNN":
            self.rnn = nn.RNN(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=self.batch_first,
                              dropout=dropout,
                              bidirectional=bidirectional)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=self.batch_first,
                               dropout=dropout,
                               bidirectional=bidirectional)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=self.batch_first,
                              dropout=dropout,
                              bidirectional=bidirectional)
        else:
            raise RuntimeError("Unknown RNN type: " + rnn_type)

        if bidirectional:
            self.combine_hidden = nn.Linear(hidden_size * 2, hidden_size)

    def _combine_directions(self, h):
        """ If the module is bidirectional, do the following transformation.
            (n_directions * n_layers, n_batch, hidden_size) ->
            (n_layers, n_batch, hidden_size)
        """
        cat_h = torch.cat([h[0 : h.size(0) : 2], h[1 : h.size(0) : 2]], 2)
        return self.combine_hidden(cat_h)

    def forward(self, inputs, seq_lens, hiddens=None):
        sorted_seq_lens, perm_idx = torch.sort(seq_lens, dim=0, descending=True)
        no_sort = seq_lens.equal(sorted_seq_lens)

        if not no_sort:
            batch_dim = 0 if self.batch_first else 1
            inputs = inputs.index_select(batch_dim, perm_idx)

        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            inputs, sorted_seq_lens, batch_first=self.batch_first)

        output, hiddens = self.rnn(packed_inputs, hiddens)

        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=self.batch_first)

        if not no_sort:
            _, unperm_idx = perm_idx.sort(0)
            output = output.index_select(batch_dim, unperm_idx)

        if self.bidirectional:
            if isinstance(hiddens, tuple):
                hiddens = tuple([self._combine_directions(h) for h in hiddens])
            else:
                hiddens = self._combine_directions(hiddens)

            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]

        return output, hiddens