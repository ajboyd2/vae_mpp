import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import xavier_truncated_normal, ACTIVATIONS


class PPAggregator(nn.Module):
    """Transforms output of a PPEncoder into a latent vector for injestion by a PPDecoder."""

    def __init__(self, method, hidden_size, latent_size, noise=True, q_z_x=torch.distributions.Laplace):
        super().__init__()

        self.noise = noise

        if method == "concat":
            self.method = self._concat
        else:
            raise ValueError

        self.mu_network = nn.Linear(hidden_size, latent_size)
        self.sigma_network = nn.Linear(hidden_size, latent_size)
        self.q_z_x = q_z_x

    def _concat(self, hidden_states, context_lengths):
        assert(len(hidden_states.shape) == 3)
        assert(len(context_lengths.shape) == 2)

        return hidden_states.gather(dim=1, index=context_lengths.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])).squeeze(1)

    def forward(self, hidden_states, context_lengths, sample_override=False):
        extracted_states = self.method(hidden_states, context_lengths)

        mu = self.mu_network(extracted_states)
        if not self.noise:
            return {
                "latent_state": mu,
                "q_z_x": None,
            }
        sigma = self.sigma_network(extracted_states) #F.softplus(self.sigma_network(extracted_states))
        sigma = F.softmax(sigma, dim=-1) * sigma.shape[-1] + 1e-6

        q_z_x = self.q_z_x(mu, sigma)

        if self.training or sample_override:
            latent_state = q_z_x.rsample()
        else:
            latent_state = mu

        return {
            "latent_state": latent_state,
            "q_z_x": q_z_x,
        }