import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):

    def __init__(
            self,
            time_embedding_dim,
            raw_frequency=False,
            raw_decay=False,
            delta_frequency=True,
            delta_decay=True,
            learnable_frequency=False,
            learnable_decay=True,
            weight_share=True
    ):
        super(TimeEmbedding, self).__init__()

        self.embedding_size = time_embedding_dim

        if (raw_frequency or raw_decay) and (delta_frequency or delta_decay):
            working_embedding_size = time_embedding_dim / 2
        else:
            working_embedding_size = time_embedding_dim

        if raw_frequency:
            raw_freq_weights = 1 / (10000 ** (torch.arange(0.0, working_embedding_size, 2) / working_embedding_size))
            if learnable_frequency:
                self.raw_freq_weights = nn.Parameter(raw_freq_weights)
            else:
                self.register_buffer("raw_freq_weights", raw_freq_weights)
        else:
            self.raw_freq_weights = None

        if raw_decay:
            raw_decay_weights = 1 / (20 + torch.rand(working_embedding_size // 2) * 30)
            if learnable_decay:
                self.raw_decay_weights = nn.Parameter(raw_decay_weights)
            else:
                self.register_buffer("raw_decay_weights", raw_decay_weights)
        else:
            self.raw_decay_weights = 0

        if delta_frequency:
            if raw_frequency and weight_share:
                self.delta_freq_weights = self.raw_freq_weights
            else:
                delta_freq_weights = 1 / (10000 ** (torch.arange(0.0, working_embedding_size, 2) / working_embedding_size))
                if learnable_frequency:
                    self.delta_freq_weights = nn.Parameter(delta_freq_weights)
                else:
                    self.register_buffer("delta_freq_weights", delta_freq_weights)
        else:
            self.delta_freq_weights = None

        if delta_decay:
            if raw_decay and weight_share:
                self.delta_decay_weights = self.raw_decay_weights
            else:
                delta_decay_weights = 1 / (20 + torch.rand(working_embedding_size // 2) * 30)
                if learnable_decay:
                    self.delta_decay_weights = nn.Parameter(delta_decay_weights)
                else:
                    self.register_buffer("delta_decay_weights", delta_decay_weights)
        else:
            self.raw_decay_weights = 0

    def forward(self, t, sample_map=None):
        '''
        sample_map is a byte tensor containing zeros in positions where values of t are for sampling
        purposes and do not represent true events.
        '''

        if (self.delta_decay_weights is None) and (self.delta_freq_weights == 0):
            return torch.cat((
                torch.cos(t * self.raw_freq_weights) * torch.exp(-t * self.raw_decay_weights.abs()),
                torch.sin(t * self.raw_freq_weights) * torch.exp(-t * self.raw_decay_weights.abs())
            ), dim=-1)
        else:
            ref_t = torch.zeros_like(t[0, :])  # slice across batches along the first time step
            deltas = []

            if sample_map is None:
                sample_map = torch.ones_like(t, dtype=torch.uint8)

            for i in range(t.shape[0]):
                deltas.append(t[i, :] - ref_t)
                ref_t = torch.where(sample_map[i, :], t[i, :], ref_t)

            d = torch.stack(deltas, dim=0)

            assert(d.shape[0] == t.shape[0])
            assert(d.shape[1] == t.shape[1])

            if (self.raw_freq_weights is None) or (self.raw_decay_weights == 0):
                return torch.cat((
                    torch.cos(d * self.delta_freq_weights) * torch.exp(-d * self.delta_decay_weights.abs()),
                    torch.sin(d * self.delta_freq_weights) * torch.exp(-d * self.delta_decay_weights.abs())
                ), dim=-1)
            else:
                return torch.cat((
                    torch.cos(d * self.delta_freq_weights) * torch.exp(-d * self.delta_decay_weights.abs()),
                    torch.sin(d * self.delta_freq_weights) * torch.exp(-d * self.delta_decay_weights.abs()),
                    torch.cos(t * self.raw_freq_weights) * torch.exp(-t * self.raw_decay_weights.abs()),
                    torch.sin(t * self.raw_freq_weights) * torch.exp(-t * self.raw_decay_weights.abs())
                ), dim=-1)