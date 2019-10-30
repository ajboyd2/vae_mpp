import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import xavier_truncated_normal, flatten, find_closest, ACTIVATIONS


class IntensityNet(nn.Module):
    """Module that transforms a set of timestamps, hidden states, and latent vector into intensity values for different channels."""

    def __init__(
        self,
        channel_embedding,
        input_size,
        hidden_size,
        num_layers,
        act_func,
        dropout,
    ):
        super().__init__()

        self.channel_embedding = channel_embedding
        self.input_size = input_size
        channel_embedding_dim = channel_embedding.weight.shape[-1]

        if isinstance(act_func, str):
            act_func = ACTIVATIONS[act_func]
        
        preprocessing_layers = [(nn.Linear(input_size, hidden_size), act_func(), nn.Dropout(p=dropout))]
        preprocessing_layers.extend([(nn.Linear(hidden_size, hidden_size), act_func(), nn.Dropout(p=dropout)) for _ in range(num_layers-1)])
        self.preprocessing_net = nn.Sequential(*flatten(preprocessing_layers))

        self.mark_net = nn.Linear(hidden_size, channel_embedding_dim)
        self.intensity_net = nn.Linear(hidden_size, 1)

    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        assert(x.shape[-1] == self.input_size)

        pre_out = self.preprocessing_net(x)

        log_mark_probs = F.linear(self.mark_net(pre_out), self.channel_embedding.weight)  # No bias by default
        log_mark_probs = F.log_softmax(log_mark_probs, dim=-1)
        
        log_intensity = self.intensity_net(pre_out)

        return {
            "log_mark_probs": log_mark_probs,
            "log_intensity": log_intensity.squeeze(-1),
        }

class PPDecoder(nn.Module):
    """Decoder module that transforms a set of marks, timestamps, and latent vector into intensity values for different channels."""

    def __init__(
        self,
        channel_embedding,
        time_embedding,
        act_func,
        num_intensity_layers,
        intensity_hidden_size,
        num_recurrent_layers,
        recurrent_hidden_size,
        dropout,
        latent_size=None,
    ):
        super().__init__()

        self.channel_embedding = channel_embedding
        self.time_embedding = time_embedding
        self.channel_embedding_size, self.time_embedding_dim = self.channel_embedding.weight.shape[-1], self.time_embedding.embedding_dim
        
        #nn.Embedding(
        #    num_embeddings=num_channels,
        #    embedding_dim=channel_embedding_size,
        #    _weight=xavier_truncated_normal((num_channels, channel_embedding_size))
        #)

        self.intensity_net = IntensityNet(
            channel_embedding=self.channel_embedding,
            input_size=latent_size + recurrent_hidden_size + self.time_embedding_dim,
            hidden_size=intensity_hidden_size,
            num_layers=num_intensity_layers,
            act_func=act_func,
            dropout=dropout,
        )

        self.recurrent_input_size = latent_size + self.channel_embedding_size + self.time_embedding_dim
        self.recurrent_net =  nn.GRU(
            input_size=self.recurrent_input_size,
            hidden_size=recurrent_hidden_size,
            num_layers=num_recurrent_layers,
            bidirectional=False,
            dropout=dropout,
            batch_first=True,
        )

        self.register_parameter(
            name="init_hidden_state",
            param=nn.Parameter(xavier_truncated_normal(size=(num_recurrent_layers, 1, recurrent_hidden_size), no_average=True))
        )

    def get_states(self, marks, timestamps, latent_state=None):
        """Produce the set of hidden states from a given set of marks, timestamps, and latent vector that can then be used to calculate intensities.
        
        Arguments:
            marks {torch.LongTensor} -- Tensor containing mark ids that correspond to channel embeddings.
            timestamps {torch.FloatTensor} -- Tensor containing times of events that correspond to the marks.

        Keyword Arguments:
            latent_state {torch.FloatTensor} -- Latent vector that [hopefully] summarizes relevant point process dynamics from a reference point pattern. (default: {None})
        
        Returns:
            torch.FloatTensor -- Corresponding hidden states that represent the history of the point process.
        """

        components = []
        components.append(self.channel_embedding(marks))
        components.append(self.time_embedding(timestamps))

        if latent_state is not None:
            components.append(latent_state.unsqueeze(1).expand(latent_state.shape[0], timestamps.shape[1], latent_state.shape[1]))

        recurrent_input = torch.cat(components, dim=-1)
        assert(recurrent_input.shape[-1] == self.recurrent_input_size)

        hidden_states, _ = self.recurrent_net(recurrent_input, self.init_hidden_state.expand(-1, recurrent_input.shape[0], -1))  # Match batch size

        return hidden_states

    def get_intensity(self, state_values, state_times, timestamps, latent_state=None):
        """Gennerate intensity values for a point process.
        
        Arguments:
            state_values {torch.FloatTensor} -- Output hidden states from `get_states` call.
            state_times {torch.FloatTensor} -- Corresponding timestamps used to generate state_values. These are the "true event times" to be compared against.
            timestamps {torch.FloatTensor} -- Times to generate intensity values for.
        
        Keyword Arguments:
            latent_state {torch.FloatTensor} -- Latent vector that [hopefully] summarizes relevant point process dynamics from a reference point pattern. (default: {None})
        
        Returns:
            [type] -- [description]
        """

        closest_dict = find_closest(sample_times=timestamps, true_times=state_times)
        padded_state_values = torch.cat((self.init_hidden_state[[-1], :, :].expand(state_values.shape[0], -1, -1), state_values), dim=1)  # To match dimensions from when closest values were found

        selected_hidden_states = padded_state_values.gather(dim=1, index=closest_dict["closest_indices"].unsqueeze(-1).expand(-1, -1, padded_state_values.shape[-1]))

        time_embedding = self.time_embedding(timestamps, state_times)

        components = [time_embedding, selected_hidden_states]
        if latent_state is not None:
            components.append(latent_state.unsqueeze(1).expand(latent_state.shape[0], timestamps.shape[1], latent_state.shape[1]))

        intensity_input = torch.cat(components, dim=-1)
        return self.intensity_net(intensity_input)
