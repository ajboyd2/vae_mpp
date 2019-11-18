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
        use_embedding_weights=False,
        factored_heads=True,
    ):
        super().__init__()

        self.channel_embedding = channel_embedding
        self.input_size = input_size
        num_channels, channel_embedding_dim = channel_embedding.weight.shape

        if isinstance(act_func, str):
            act_func = ACTIVATIONS[act_func]
        
        preprocessing_layers = [(nn.Linear(input_size, hidden_size), act_func(), nn.Dropout(p=dropout))]
        preprocessing_layers.extend([(nn.Linear(hidden_size, hidden_size), act_func(), nn.Dropout(p=dropout)) for _ in range(num_layers-1)])
        self.preprocessing_net = nn.Sequential(*flatten(preprocessing_layers))

        self.factored_heads = factored_heads
        if factored_heads:
            self.mark_net = nn.Linear(hidden_size, channel_embedding_dim if use_embedding_weights else num_channels)
            self.intensity_net = nn.Linear(hidden_size, 1)
            self.use_embedding_weights = use_embedding_weights
        else:
            print("NON FACTORED HEADS")
            self.mark_net = nn.Sequential(nn.Linear(hidden_size, num_channels), ACTIVATIONS["softplus"]())

    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        assert(x.shape[-1] == self.input_size)

        pre_out = self.preprocessing_net(x)

        mark_net_out = self.mark_net(pre_out)
        if self.factored_heads:
            if self.use_embedding_weights:
                log_mark_probs = F.linear(self.mark_net(pre_out), self.channel_embedding.weight)  # No bias by default
            else:
                log_mark_probs = mark_net_out

            log_mark_probs = F.log_softmax(log_mark_probs, dim=-1)
            log_intensity = self.intensity_net(pre_out)
            all_log_mark_intensities = log_mark_probs + log_intensity
            total_intensity = log_intensity.exp().squeeze(dim=-1)
        else:
            all_log_mark_intensities = torch.log(mark_net_out + 1e-12)
            total_intensity = mark_net_out.sum(dim=-1)

        return {
            "all_log_mark_intensities": all_log_mark_intensities,
            "total_intensity": total_intensity,
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
        factored_heads=True,
        estimate_init_state=True,
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
            input_size=latent_size + self.time_embedding_dim + recurrent_hidden_size,
            hidden_size=intensity_hidden_size,
            num_layers=num_intensity_layers,
            act_func=act_func,
            dropout=dropout,
            factored_heads=factored_heads,
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

        self.latent_size = latent_size
        self.num_recurrent_layers = num_recurrent_layers
        self.recurrent_hidden_size = recurrent_hidden_size
        self.estimate_init_state = (latent_size is not None) and estimate_init_state
        if self.estimate_init_state:
            print("ESTIMATING INITIAL STATE")
            self.init_hidden_state_network = nn.Sequential(
                nn.Linear(latent_size, num_recurrent_layers * recurrent_hidden_size),
                #ACTIVATIONS["gelu"](),
            )
        else:
            print("NOT ESTIMATING INITIAL STATE")
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

        if self.estimate_init_state:
            init_hidden_state = self.init_hidden_state_network(latent_state).view(-1, self.num_recurrent_layers, self.recurrent_hidden_size)
            init_hidden_state = torch.transpose(init_hidden_state, 0, 1)
        else:
            init_hidden_state = self.init_hidden_state.expand(-1, recurrent_input.shape[0], -1)  # Match batch size
        hidden_states, _ = self.recurrent_net(recurrent_input, init_hidden_state) 

        return torch.cat((init_hidden_state[-1, :, :].unsqueeze(1), hidden_states), dim=1)

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

        padded_state_values = state_values #torch.cat((self.init_hidden_state[[-1], :, :].expand(state_values.shape[0], -1, -1), state_values), dim=1)  # To match dimensions from when closest values were found

        selected_hidden_states = padded_state_values.gather(dim=1, index=closest_dict["closest_indices"].unsqueeze(-1).expand(-1, -1, padded_state_values.shape[-1]))

        time_embedding = self.time_embedding(timestamps, state_times)

        components = [time_embedding, selected_hidden_states]
        if latent_state is not None:
            components.append(latent_state.unsqueeze(1).expand(latent_state.shape[0], timestamps.shape[1], latent_state.shape[1]))

        intensity_input = torch.cat(components, dim=-1)
        return self.intensity_net(*components)#intensity_input)
