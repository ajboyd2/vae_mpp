import torch
from torch import nn

ACTIVATIONS = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'log': torch.log,
    'identity': lambda x: (lambda y: y)(x)
}

def truncated_normal(size, scale=1, limit=2):
    """Samples a tensor from an approximately truncated normal tensor.
    
    Arguments:
        size {tuple of ints} -- Size of desired tensor
    
    Keyword Arguments:
        scale {int} -- Standard deviation of normal distribution (default: {1})
        limit {int} -- Number of standard deviations to truncate at (default: {2})
    
    Returns:
        torch.FloatTensor -- A truncated normal sample of requested size
    """
    return torch.fmod(torch.randn(size),limit) * scale

def xavier_truncated_normal(size, limit=2, no_average=False):
    """Samples from a truncated normal where the standard deviation is automatically chosen based on size."""
    if len(size) == 1 or no_average:
        n_avg = size[-1]
    else:
        n_in, n_out = size[-2], size[-1]
        n_avg = (n_in + n_out) / 2
    
    return truncated_normal(size, scale=(1/n_avg)**0.5, limit=2)

def flatten(list_of_lists):
    """Turn a list of lists (or any iterable) into a flattened list."""
    return [item for sublist in list_of_lists for item in sublist]

def find_closest(sample_times, true_times):
    """For each value in sample_times, find the values and associated indices in true_times that are 
    closest and strictly less than.
    
    Arguments:
        sample_times {torch.FloatTensor} -- Contains times that we want to find values closest but not over them in true_times
        true_times {torch.FloatTensor} -- Will take the closest times from here compared to sample_times
    
    Returns:
        dict -- Contains the closest values and corresponding indices from true_times.
    """
    # Pad true events with zeros (if a value in t is smaller than all of true_times, then we have it compared to time=0)
    padded_true_times =  torch.cat((true_times[..., [0]]*0, true_times), dim=-1)  

    # Format true_times to have all values compared against all values of t
    size = padded_true_times.shape
    expanded_true_times = padded_true_times.unsqueeze(-1).expand(*size, sample_times.shape[-1])  
    expanded_true_times = expanded_true_times.permute(*list(range(len(size)-1)), -1, -2)

    # Find out which true event times happened after which times in t, then mask them out
    mask = expanded_true_times < sample_times.unsqueeze(-1)
    adjusted_expanded_true_times = torch.where(mask, expanded_true_times, -expanded_true_times*float('inf'))

    # Find the largest, unmasked values. These are the closest true event times that happened prior to the times in t.
    closest_values, closest_indices = adjusted_expanded_true_times.max(dim=-1)

    return {
        "closest_values": closest_values,
        "closest_indices": closest_indices,
    }