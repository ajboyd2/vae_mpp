import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):

    def __init__(self, time_embedding_dim, learnable=False):
        super(TimeEmbedding, self).__init__()

        self.embedding_size = time_embedding_dim

        weights = 1 / (10000 ** (torch.arange(0.0, time_embedding_dim, 2) / time_embedding_dim))
        if learnable:
            self.weights = nn.Parameter(weights)
        else:
            self.register_buffer("weights", weights)

    def forward(self, t):
        return torch.cat((
            torch.cos(t * self.weights),
            torch.sin(t * self.weights)
        ), dim=-1)