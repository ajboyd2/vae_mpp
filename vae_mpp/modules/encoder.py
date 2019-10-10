import torch
import torch.nn as nn
import numpy as np

from .utils import xavier_truncated_normal, ACTIVATIONS

class PPEncoder(nn.Module):
    """Encoder module that transforms a sequence of referential marks and times to a set of hidden states.
    Couple with an PPAdapter to transform output into a latent vector for the PPDecoder.
    
    """

    def __init__(
        self, 
        channel_embedding,
        time_embedding,
        hidden_size,
        bidirectional,
        num_recurrent_layers,
        dropout,
    ):
        self.channel_embedding = channel_embedding
        self.time_embedding = time_embedding
        self.channel_embedding_size, self.time_embedding_dim = self.channel_embedding._weight.shape[-1], self.time_embedding.embedding_dim

        recurrent_net_args = {
            "input_size": self.channel_embedding_size + self.time_embedding_dim,
            "hidden_size": hidden_size,
            "num_layers": num_recurrent_layers,
            "batch_first": True,
            "bidirectional": False,  # Need to keep separate networks if bidir=True due to potential padding issues
            "dropout": dropout,
        }
        self.forward_recurrent_net = nn.GRU(**recurrent_net_args)
        self.register_parameter(
            name="forward_init_hidden_state",
            param=xavier_truncated_normal(size=(num_recurrent_layers, 1, hidden_size), no_average=True)
        )

        self.bidirectional = bidirectional
        if bidirectional:
            self.backward_recurrent_net = nn.GRU(**recurrent_net_args)
            self.register_parameter(
                name="backward_init_hidden_state",
                param=xavier_truncated_normal(size=(num_recurrent_layers, 1, hidden_size), no_average=True)
            )
        else:
            self.backward_recurrent_net = None

    def forward(self, forward_marks, forward_timestamps, backward_marks=None, backward_timestamps=None):
        steps = [(forward_marks, forward_timestamps, self.forward_recurrent_net, self.forward_init_hidden_state)]
        if self.bidirectional:
            assert(backward_marks is not None and backward_timestamps is not None)
            steps.append((backward_marks, backward_timestamps, self.backward_recurrent_net, self.backward_init_hidden_state))
        
        hidden_states = []
        for marks, timestamps, recurrent_net, init_hidden_state in steps:
            mark_embedding = self.channel_embedding(marks)
            time_embedding = self.time_embedding(timestamps)
            recurrent_input = torch.cat((mark_embedding, time_embedding), dim=-1)

            hidden_states = recurrent_net(recurrent_input)[0]  # output is a tuple, first element are all hidden states for last layer second is last hidden state for all layers

        return torch.cat(hidden_states, dim=-1)
