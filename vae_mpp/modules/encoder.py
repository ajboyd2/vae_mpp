import torch
import torch.nn as nn
import numpy as np

_activations = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'log': torch.log,
    'identity': lambda x: (lambda y: y)(x)
}


class PPEncoder(nn.Module):

    def __init__(self,
                 event_embedding_dim,
                 time_embedding_dim,
                 latent_size,
                 bidirectional,
                 rnn_layers,
                 aggregation_style,
                 var_act='identity',
                 dropout=0.0
    ):
        super(PPEncoder, self).__init__()

        self.latent_size = latent_size
        self.hidden_size = latent_size * (1 if bidirectional else 2)

        self.rnn = nn.GRU(
            input_size=event_embedding_dim + time_embedding_dim,
            hidden_size=self.hidden_size,
            bidirectional=bidirectional,
            bias=True,
            batch_first=False,
            num_layers=rnn_layers,
            dropout=dropout)

        if aggregation_style == "concatenation":
            self.aggregation = self._concatenation
        elif aggregation_style == "mean":
            self.aggregation = self._mean
        elif aggregation_style == "max":
            self.aggregation = self._max
        elif aggregation_style == "attention":
            raise NotImplementedError
        else:
            raise LookupError("aggregation_style '{}' not supported.".format(aggregation_style))

        self.var_act = _activations[var_act]

    def _concatenation(self, output, mask=None):
        if mask is None:
            return torch.cat((output[-1, :, :self.hidden_size], output[0, :, self.hidden_size:]), dim=-1)
        raise NotImplementedError


    def _mean(self, output, mask=None):
        if mask is None:
            return torch.mean(output, dim=0)

        return torch.where(
            mask.unsqueeze(-1).expand(-1, -1, output.shape[-1]),
            output,
            torch.zeros_like(output)
        ).sum(dim=0) / mask.sum(dim=0).float()

    def _max(self, output, mask=None):
        if mask is None:
            return torch.max(output, dim=0)[0]

        return torch.where(
            mask.unsqueeze(-1).expand(-1, -1, output.shape[-1]),
            output,
            torch.ones_like(output) * (-np.inf)
        ).max(dim=0)[0]

    def _attention(selfs, output, mask=None):
        raise NotImplemented

    def forward(self, time_embed, mark_embed, mask=None):
        # batch.shape = (sequence_length, batch_size, mark_embedding_dim+time_embedding_dim)
        batch = torch.cat((time_embed, mark_embed), dim=-1)
        output, _ = self.rnn(batch)
        pooled_output = self.aggregation(output, mask)
        mu, sigma = pooled_output[:, ::2], self.var_act(pooled_output[:, 1::2])
        return mu, sigma
