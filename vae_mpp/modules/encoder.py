import torch
import torch.nn as nn

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
                 var_act,
                 dropout=0.0
    ):

        self.latent_size = latent_size
        hidden_size = latent_size * 2

        self.rnn = nn.GRU(
            input_size=event_embedding_dim + time_embedding_dim,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            bias=True,
            num_layers=rnn_layers,
            dropout=dropout)

        if aggregation_style == "concatenation":
            self.aggregation = lambda output: torch.cat((output[-1, :, :hidden_size], output[0, :, hidden_size:]),
                                                        dim=-1)
        elif aggregation_style == "mean":
            self.aggregation = lambda output: torch.mean(output, dim=0)
        elif aggregation_style == "max":
            self.aggregation = lambda output: torch.max(output, dim=0)
        elif aggregation_style == "attention":
            raise NotImplemented
        else:
            raise LookupError("aggregation_style '{}' not supported.".format(aggregation_style))

        self.var_act = _activations[var_act]

    def forward(self, batch):
        # batch.shape = (sequence_length, batch_size, embedding_dim+1)
        output, _ = self.rnn(batch)
        pooled_output = self.aggregation(output)
        mu, sigma = pooled_output[:, ::2], self.var_act(pooled_output[:, 1::2])
        return mu, sigma
