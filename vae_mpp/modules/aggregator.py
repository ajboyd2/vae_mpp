import torch
import torch.nn as nn
import numpy as np

from .utils import xavier_truncated_normal, ACTIVATIONS


class PPAggregator(nn.Module):
    """Transforms output of a PPEncoder into a latent vector for injestion by a PPDecoder."""

    def __init__(self, method, hidden_size, noise=True):
        super().__init__()

        self.noise = noise

        if method == "concat":
            self.method = self._concat
        else:
            raise ValueError

        self.mu_network = nn.Linear(hidden_size, hidden_size)
        self.log_sigma_network = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, context_lengths, sample_override=False):
        extracted_states = self.method(hidden_states)

        mu, log_sigma = self.mu_network(extracted_states), self.log_sigma_network(extracted_states)

        if self.training or sample_override:
            latent_state = torch.randn_like(mu) * torch.exp(log_sigma) + mu
        else:
            latent_state = mu

        return {
            "latent_state": latent_state,
            "mu": mu,
            "log_sigma": log_sigma,
        }