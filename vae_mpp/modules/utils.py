import torch

def truncated_normal(size, scale=1, limit=2):
    return torch.fmod(torch.randn(size),limit) * scale

def xavier_truncated_normal(size, limit=2):
    if len(size) == 1:
        n_avg = size
    else:
        n_in, n_out = size[-2], size[-1]
        n_avg = (n_in + n_out) / 2
    
    return truncated_normal(size, scale=(1/n_avg)**0.5, limit=2)

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]
