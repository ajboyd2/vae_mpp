import torch


ACTIVATIONS = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'log': torch.log,
    'identity': lambda x: (lambda y: y)(x)
}

def truncated_normal(size, scale=1, limit=2):
    return torch.fmod(torch.randn(size),limit) * scale

def xavier_truncated_normal(size, limit=2, no_average=False):
    if len(size) == 1 or no_average:
        n_avg = size[-1]
    else:
        n_in, n_out = size[-2], size[-1]
        n_avg = (n_in + n_out) / 2
    
    return truncated_normal(size, scale=(1/n_avg)**0.5, limit=2)

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def find_closest(sample_times, true_times):
    """For each value in sample_times, find the values and associated indices in true_times that are 
    closest and strictly less than.
    
    Arguments:
        sample_times {[type]} -- [description]
        true_times {[type]} -- [description]
    
    Returns:
        [type] -- [description]
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