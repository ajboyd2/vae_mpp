import torch

def truncated_normal(size, scale=1, limit=2):
    return torch.fmod(torch.randn(size),limit) * scale