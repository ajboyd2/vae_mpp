import torch
import torch.nn as nn

from vae_mpp.modules import PPDecoder, PPEncoder, TimeEmbedding
from vae_mpp.models import Model


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


@Model.register('vae')
class MPPVae(nn.Module):

    def __init__(self,
                 num_events,
                 event_embedding_size,
                 time_embedding_size,
                 raw_time,
                 delta_time,
                 latent_size,
                 enc_bidirectional,
                 enc_rnn_layers,
                 enc_aggregation_style,
                 divergence,
                 dec_hidden_size,
                 intensity_layer_size,
                 intensity_num_layers,
                 intensity_act_func,
                 layer_norm,
                 **kwargs):
        super(MPPVae, self).__init__()

        self.num_events = num_events
        self.event_embedding_size = event_embedding_size
        self.time_embedding_size = time_embedding_size

        self.event_embeddings = nn.Embedding(num_events, event_embedding_size)

        if not (raw_time or delta_time):
            raise AttributeError("At most, only one of 'raw_time' or 'delta_time' can be False.")

        self.time_embeddings = TimeEmbedding(
            time_embedding_dim=time_embedding_size,
            raw_frequency=raw_time,
            raw_decay=False,
            delta_frequency=delta_time,
            delta_decay=delta_time,
            learnable_frequency=False,
            learnable_decay=True,
            weight_share=True
        )

        self.encoder = PPEncoder(
            event_embedding_dim=event_embedding_size,
            time_embedding_dim=time_embedding_size,
            latent_size=latent_size,
            bidirectional=enc_bidirectional,
            rnn_layers=enc_rnn_layers,
            aggregation_style=enc_aggregation_style,
            var_act='identity',
            dropout=0.0
        )

        self.decoder = PPDecoder(
            num_events=num_events,
            hidden_size=dec_hidden_size,
            event_embedding_size=event_embedding_size,
            time_embedding_size=time_embedding_size,
            latent_size=latent_size,
            intensity_layer_size=intensity_layer_size,
            intensity_num_layers=intensity_num_layers,
            intensity_act_func=intensity_act_func,
            layer_norm=layer_norm
        )

        if divergence == "kl_div":
            self.div_func = self.kl_div
        elif divergence == "mmd_div":
            self.div_func = self.mmd_div
        else:
            raise LookupError("Divergence type '{}' is not supported.".format(divergence))

        self.register_buffer("eye", torch.eye(num_events))

    def forward(self, times, marks, enc_times=None, enc_marks=None, padding_mask=None, mc_samples=100, T=None, **kwargs):

        times = times.t().contiguous()
        marks = marks.t().contiguous()

        if enc_times is None or enc_marks is None:
            enc_times = times
            enc_marks = marks
        else:
            enc_times = enc_times.t().contiguous()
            enc_marks = enc_marks.t().contiguous()

        if padding_mask is not None:
            padding_mask = padding_mask.t().contiguous()

        # encode
        mu, log_sigma_sq = self.encoder(
            time_embed=self.time_embeddings(enc_times.unsqueeze(-1)),
            mark_embed=self.event_embeddings(enc_marks),
            mask=padding_mask
        )

        # sample latent state
        zeros = torch.zeros_like(mu)
        epsilon = torch.normal(zeros, zeros + 1)  # standard normal
        latent_state = epsilon * torch.exp(0.5 * log_sigma_sq) + mu

        # decode
        _, batch_size = times.shape

        if T is None:
            T = times.max(dim=0)[0]
        else:
            T = torch.ones_like(times.max(dim=0)[0]) * T  # T is assumed to be a float
        u = T.unsqueeze(0) * torch.rand_like(T.unsqueeze(0).expand(mc_samples, batch_size))

        if padding_mask is None:
            padding_mask = 1

        mask_unsorted = torch.cat((
            torch.ones_like(times, dtype=torch.uint8) + (1 - padding_mask),
            torch.zeros_like(u, dtype=torch.uint8)
        ), dim=0)  # ones for original times, zeros for sampled ones, twos for padding

        times_unsorted = torch.cat((
            times,
            u
        ), dim=0)

        marks_unsorted = torch.cat((
            marks,
            torch.zeros_like(marks[0, :]).unsqueeze(0).expand(mc_samples, -1)
        ), dim=0)

        times_sorted, indices = torch.sort(times_unsorted, dim=0)
        marks_sorted = torch.gather(marks_unsorted, dim=0, index=indices)
        mask_sorted = torch.gather(mask_unsorted, dim=0, index=indices)

        original_events_mask = mask_sorted == 1
        sample_events_mask = mask_sorted == 0

        time_embed = self.time_embeddings(times_sorted.unsqueeze(-1), mask_sorted)
        mark_embed = self.event_embeddings(marks_sorted)

        output = self.decoder(
            time_embed=time_embed,
            mark_embed=mark_embed,
            mask=original_events_mask,
            latent_state=latent_state
        )

        output["mu"] = mu
        output["log_sigma_sq"] = log_sigma_sq

        # compute negative log likelihood loss
        log_sum = torch.where(
            original_events_mask.unsqueeze(-1).expand(-1, -1, self.num_events),
            output["log_intensities"] + output["log_probs"],
            torch.zeros_like(output["log_probs"])
        ).gather(index=marks_sorted.unsqueeze(-1), dim=-1).squeeze().sum(dim=0)

        int_approx = T * torch.where(
            sample_events_mask,
            torch.exp(output["log_intensities"].squeeze()),
            torch.zeros_like(output["log_intensities"].squeeze())
        ).sum(dim=0) / mc_samples

        output["all_times"] = times_sorted
        output["log_sum"] = log_sum
        output["int_approx"] = int_approx
        output["nll"] = -1 * (log_sum - int_approx)  # negative log likelihood

        # compute divergence
        output["div"] = self.div_func(output)

        output["loss"] = output["nll"] + output["div"]

        return output

    def kl_div(self, output):
        mu = output["mu"]
        log_sigma_sq = output["log_sigma_sq"]
        return (0.5 * (mu ** 2 + torch.exp(log_sigma_sq) - log_sigma_sq - 1).sum(dim=1)).mean()

    def mmd_div(self, output):
        z = output["latent_state"]
        true_samples = torch.rand_like(z)
        return compute_mmd(true_samples, z)