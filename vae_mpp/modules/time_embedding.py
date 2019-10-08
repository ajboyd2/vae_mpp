import math
import torch
import torch.nn as nn

from .utils import truncated_normal


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal-based encodings. Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Attributes:
        weight {torch.tensor} -- Weight tensor to multiply with incoming times
    """
    def __init__(self, embedding_dim, learnable=False, random=False):
        super().__init__()

        self.random = random
        if random:
            weight = truncated_normal(size=embedding_dim//2, scale=0.01, limit=2)
        else:
            weight = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
    
        if learnable:
            self.register_parameter('weight', weight)
        else:
            self.register_buffer('weight', weight)

    def forward(self, t):
        sin_emb = torch.sin(t * self.weight)
        cos_emb = torch.cos(t * self.weight)
        return torch.cat((sin_emb, cos_emb), dim=-1)

class ExponentialEmbedding(SinusoidalEmbedding):

    def __init__(self, embedding_dim, learnable=True, random=True):
        super().__init__(
            embedding_dim=embedding_dim * (2 if random else 1),  #  If used in conjunction with SinusoidalEmbeddings, should follow the torch.cat(...) pattern 
            learnable=learnable,
            random=random,
        )

    def forward(self, t):
        embedding = torch.exp(-t * self.weight.abs())
        if self.random:
            return embedding
        else:
            return torch.cat((embedding, embedding), dim=-1)

class SinExpEmbedding(nn.Module):

    def __init__(self, embedding_dim, use_sinusoidal=True, use_exponential=False, sin_rand=False, exp_rand=False):
        assert(use_sinusoidal or use_exponential)
        
        if use_sinusoidal:
            self.sin_embed = SinusoidalEmbedding(
                embedding_dim=embedding_dim,
                learnable=sin_rand,  # For now, assume if the weights are randomly initialized, then they are also learnable
                random=sin_rand,
            )
        else:
            self.sin_embed = None
        
        if use_exponential:
            self.exp_embed = ExponentialEmbedding(
                embedding_dim=embedding_dim,
                learnable=exp_rand,  # For now, assume if the weights are randomly initialized, then they are also learnable
                random=exp_rand,
            )
        else:
            self.exp_embed = None
        
    def forward(self, t):
        if self.sin_embed:
            sin_embedding = self.sin_embed(t)
        else:
            sin_embedding = 1

        if self.exp_embed:
            exp_embedding = self.exp_embed(t)
        else:
            exp_embedding = 1

        return sin_embedding * exp_embedding

class TemporalEmbedding(nn.Module):

    def __init__(self, embedding_dim, use_raw_time=True, use_delta_time=False, learnable_delta_weights=True):
        assert(use_raw_time or use_delta_time)
        num_components = 2 if use_raw_time and use_delta_time else 1

        if use_raw_time:
            self.raw_time_embed = SinExpEmbedding(
                embedding_dim=embedding_dim//num_components,
                use_sinusoidal=True,
                use_exponential=False,
                sin_rand=False,
                exp_rand=False,
            )
        else:
            self.raw_time_embed = None

        if use_delta_time:
            self.delta_time_embed = SinExpEmbedding(
                embedding_dim=embedding_dim//num_components,
                use_sinusoidal=True,
                use_exponential=True,
                sin_rand=False,
                exp_rand=True,
            )
        else:
            self.delta_time_embed = None

    @staticmethod
    def compute_deltas(t, true_times):
        padded_true_times =  torch.cat((true_times[..., [0]]*0, true_times), dim=-1)  # Pad true events with zeros
        size = padded_true_times.shape
        expanded_true_times = padded_true_times.unsqueeze(-1).expand(*size, t.shape[-1])  # Make tensor to compare all values of t to all values of padded_true_times
        expanded_true_times = expanded_true_times.permute(*list(range(len(size)-1)), -1, -2)  # Remove from one of the padded dimensions

        mask = expanded_true_times < t.unsqueeze(-1)
        adjusted_expanded_true_times = torch.where(mask, expanded_true_times, -expanded_true_times*float('inf'))
        past_t, _ = adjusted_expanded_true_times.max(dim=-1)

        return t - past_t

    def forward(self, t, true_times=None):
        # true_times are the timestamps of events that have actually happened
        # this is necessary as we sometimes sample for times that don't actually happen
        if true_times is None:
            true_times = t

        embeddings = []
        if self.raw_time_embed:
            embeddings.append(self.raw_time_embed(t.unsqueeze(-1)))

        if self.delta_time_embed:
            delta_t = TemporalEmbedding.compute_deltas(t, true_times)
            embeddings.append(self.delta_time_embed(delta_t.unsqueeze(-1)))

        return torch.cat(embeddings, dim=-1)
