import math
import torch
import torch.nn as nn

from .utils import xavier_truncated_normal, find_closest


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal-based encodings. Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Attributes:
        weight {torch.tensor} -- Weight tensor to multiply with incoming times
    """
    def __init__(self, embedding_dim, learnable=False, random=False, max_period=10000.0):
        super().__init__()

        self.random = random
        if random:
            weight = xavier_truncated_normal(size=embedding_dim//2, limit=2)
        else:
            weight = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(max_period) / embedding_dim))
    
        if learnable:
            self.register_parameter('weight', nn.Parameter(weight))
        else:
            self.register_buffer('weight', weight)

    def forward(self, t):
        sin_emb = torch.sin(t * self.weight)
        cos_emb = torch.cos(t * self.weight)
        return torch.cat((sin_emb, cos_emb), dim=-1)

class ExponentialEmbedding(SinusoidalEmbedding):
    """Exponential decay based embedding.
    
    Attributes:
        weight {torch.tensor} -- Weight tensor to multiply with incoming times
    """

    def __init__(self, embedding_dim, learnable=True, random=True, max_period=10000.0):
        super().__init__(
            embedding_dim=embedding_dim * (2 if random else 1),  #  If used in conjunction with SinusoidalEmbeddings, should follow the torch.cat(...) pattern 
            learnable=learnable,
            random=random,
            max_period=max_period,
        )

    def forward(self, t):
        embedding = torch.exp(-t * self.weight.abs())
        if self.random:
            return embedding
        else:
            return torch.cat((embedding, embedding), dim=-1)

class SinExpEmbedding(nn.Module):
    """Houses a SinusoidalEmbedding and/or ExponentialEmbedding, where the results are element-wise multiplied together.
    
    Attributes:
        sin_embed {SinusoidalEmbedding} -- An optional module containing sinusoidal embeddings
        exp_embed {ExponentialEmbedding} -- An optional module containing exponential embeddings
    """

    def __init__(self, embedding_dim, use_sinusoidal=True, use_exponential=False, sin_rand=False, exp_rand=False, max_period=10000.0):
        super().__init__()

        assert(use_sinusoidal or use_exponential)
        
        if use_sinusoidal:
            self.sin_embed = SinusoidalEmbedding(
                embedding_dim=embedding_dim,
                learnable=sin_rand,  # For now, assume if the weights are randomly initialized, then they are also learnable
                random=sin_rand,
                max_period=max_period,
            )
        else:
            self.sin_embed = None
        
        if use_exponential:
            self.exp_embed = ExponentialEmbedding(
                embedding_dim=embedding_dim,
                learnable=exp_rand,  # For now, assume if the weights are randomly initialized, then they are also learnable
                random=exp_rand,
                max_period=max_period,
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
    """Top level embedding that allows for sin|exp based embeddings with raw times and/or time deltas.
    
    Attributes:
        raw_time_embed {SinExpEmbedding} -- Embedding module that embeds raw timestamps
        delta_time_embed {SinExpEmbedding} -- Embedding module that embeds timestamps based on the difference between them and the nearest true event times
        embedding_dim {int} -- Size of the output embedding dimension
    """

    def __init__(
        self, 
        embedding_dim, 
        use_raw_time=True, 
        use_delta_time=False, 
        learnable_delta_weights=True, 
        max_period=10000.0,
    ):
        super().__init__()

        assert(use_raw_time or use_delta_time)
        num_components = 2 if use_raw_time and use_delta_time else 1

        if use_raw_time:
            if embedding_dim == 1:
                self.raw_time_embed = lambda x: x#.unsqueeze(-1)
            else:
                self.raw_time_embed = SinExpEmbedding(
                    embedding_dim=embedding_dim//num_components,
                    use_sinusoidal=True,
                    use_exponential=False,
                    sin_rand=False,
                    exp_rand=False,
                    max_period=max_period,
                )
        else:
            self.raw_time_embed = None

        if use_delta_time:
            if embedding_dim == 1:
                self.delta_time_embed = lambda x: x#.unsqueeze(-1)
            else:
                self.delta_time_embed = SinExpEmbedding(
                    embedding_dim=embedding_dim//num_components,
                    use_sinusoidal=True,
                    use_exponential=True,
                    sin_rand=False,
                    exp_rand=True,
                    max_period=max_period,
                )
        else:
            self.delta_time_embed = None
        
        self.embedding_dim = embedding_dim

    def forward(self, t, true_times=None):
        # true_times are the timestamps of events that have actually happened
        # this is necessary as we sometimes sample for times that don't actually happen
        if true_times is None:
            true_times = t

        embeddings = []
        if self.raw_time_embed is not None:
            embeddings.append(self.raw_time_embed(t.unsqueeze(-1)))

        if self.delta_time_embed is not None:
            closest_dict = find_closest(sample_times=t, true_times=true_times)
            delta_t = t - closest_dict["closest_values"]
            embeddings.append(self.delta_time_embed(delta_t.unsqueeze(-1)))

        embeddings = torch.cat(embeddings, dim=-1)
        return embeddings
