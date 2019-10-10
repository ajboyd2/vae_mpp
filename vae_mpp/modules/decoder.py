import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import xavier_truncated_normal, flatten, find_closest

ACTIVATIONS = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'identity': lambda x: (lambda y: y)(x)
}

class IntensityNet(nn.Module):

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
        channel_embedding_dim = channel_embedding._weight.shape[-1]

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
        mark_logits = F.linear(self.mark_net(pre_out), self.channel_embedding._weight)  # No bias by default
        intensity_logit = self.intensity_net(pre_out)

        return {
            "mark_logits": mark_logits,
            "intensity_logit": intensity_logit,
        }

class PPDecoder(nn.Module):

    def __init__(
        self,
        num_channels,
        channel_embedding_size,
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

        self.num_channels = num_channels
        self.channel_embeddings = nn.Embedding(
            num_embeddings=num_channels,
            embedding_dim=channel_embedding_size,
            _weight=xavier_truncated_normal((num_channels, channel_embedding_size))
        )

        self.time_embedding = time_embedding
        self.time_embedding_dim = self.time_embedding.embedding_dim

        self.intensity_net = IntensityNet(
            channel_embedding=self.channel_embeddings,
            input_size=latent_size + recurrent_hidden_size + self.time_embedding_dim,
            hidden_size=intensity_hidden_size,
            num_layers=num_intensity_layers,
            act_func=act_func,
            dropout=dropout,
        )

        self.recurrent_input_size = latent_size + channel_embedding_size + self.time_embedding_dim
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
            param=xavier_truncated_normal(size=(num_recurrent_layers, 1, recurrent_hidden_size), no_average=True)
        )

    def get_states(self, marks, timestamps, latent_state=None):

        components = []
        components.append(self.channel_embeddings(marks))
        components.append(self.time_embedding(timestamps))

        if latent_state:
            components.append(latent_state.unsqueeze(0).expand(timestamps.shape[0], *latent_state.shape))
        
        recurrent_input = torch.cat(components, dim=-1)
        assert(recurrent_input.shape[-1] == self.recurrent_input_size)

        hidden_states, _ = self.recurrent_net(recurrent_input, self.init_hidden_state.expand(-1, recurrent_input.shape[0], -1))  # Match batch size

        return hidden_states

    def get_intensity(self, state_values, state_times, timestamps, latent_state):

        closest_dict = find_closest(sample_times=timestamps, true_times=state_times)
        padded_state_values = torch.cat((self.init_hidden_state[[-1], :, :].expand(state_values.shape[0], -1, -1), state_values), dim=1)  # To match dimensions from when closest values were found

        selected_hidden_states = padded_state_values.gather(dim=1, index=closest_dict["closest_indices"].unsqueeze(-1).expand(-1, -1, padded_state_values.shape[-1]))

        time_embedding = self.time_embedding(timestamps)

        components = [time_embedding, selected_hidden_states]
        if latent_state is not None:
            components.append(latent_state)
        
        intensity_input = torch.cat(components, dim=-1)
        return self.intensity_net(intensity_input)
