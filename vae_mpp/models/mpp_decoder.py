import torch
import torch.nn as nn

from vae_mpp.modules import PPDecoder, TimeEmbedding
from vae_mpp.models import Model


@Model.register('decoder_only')
class MPPDecoder(nn.Module):

    def __init__(self,
                 num_events,
                 event_embedding_size,
                 time_embedding_size,
                 raw_time,
                 delta_time,
                 hidden_size,
                 latent_size,
                 intensity_layer_size,
                 intensity_num_layers,
                 intensity_act_func,
                 layer_norm,
                 **kwargs):
        super(MPPDecoder, self).__init__()

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

        self.decoder = PPDecoder(
            num_events=num_events,
            hidden_size=hidden_size,
            event_embedding_size=event_embedding_size,
            time_embedding_size=time_embedding_size,
            latent_size=latent_size,
            intensity_layer_size=intensity_layer_size,
            intensity_num_layers=intensity_num_layers,
            intensity_act_func=intensity_act_func,
            layer_norm=layer_norm
        )

        self.register_buffer("eye", torch.eye(num_events))

    def forward(self, times, marks, padding_mask=None, mc_samples=100, T=None, **kwargs):

        times = times.t().contiguous()
        marks = marks.t().contiguous()
        if padding_mask is not None:
            padding_mask = padding_mask.t().contiguous()

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

        time_embed = self.time_embeddings(times_sorted.unsqueeze(-1))
        mark_embed = self.event_embeddings(marks_sorted)

        output = self.decoder(time_embed, mark_embed, original_events_mask)

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
        output["loss"] = -1 * (log_sum - int_approx)  # loss is negative log likelihood
        return output