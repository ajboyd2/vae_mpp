import torch

from datetime import datetime


def print_log(*args):
    print("[{}]".format(datetime.now()), *args)

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input)  # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

#def kl_div(mu, log_var):
#    return (0.5 * (mu.pow(2) + torch.exp(log_var) - log_var - 1).sum(dim=1)).mean()
#def kl_div(mu, sigma):
#    batch_size = mu.shape[0]
#    return (0.5 * (mu.pow(2) + sigma.pow(2) - torch.log(1e-8 + sigma.pow(2)) - 1).sum(dim=1)).sum() / batch_size

def kl_div(d1, d2, K=100):
    """Computes closed-form KL if available, else computes a MC estimate."""
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        return torch.distributions.kl_divergence(d1, d2)
    else:
        samples = d1.rsample(torch.Size([K]))
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)

def mmd_div(z):
    raise NotImplementedError
#    true_samples = torch.randn_like(z)
#    return compute_mmd(true_samples, z)

