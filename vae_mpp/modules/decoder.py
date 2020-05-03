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
        zero_inflated=False,
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
        self.mark_net = nn.Linear(hidden_size, channel_embedding_dim if use_embedding_weights else num_channels)
        self.use_embedding_weights = use_embedding_weights
        if use_embedding_weights:
            print("USING EMBEDDING WEIGHTS")
        else:
            print("NOT USING EMBEDDING WEIGHTS")
            
        if factored_heads:
            print("FACTORED HEADS")
            self.intensity_net = nn.Linear(hidden_size, 1)
        else:
            print("NON FACTORED HEADS")
            self.softplus = nn.Softplus()

        self.zero_inflated = zero_inflated
        if zero_inflated:
            self.zero_prob_net = nn.Sequential(nn.Linear(hidden_size, num_channels), nn.Sigmoid())

    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        assert(x.shape[-1] == self.input_size)
        zero_probs = None

        pre_out = self.preprocessing_net(x)

        mark_net_out = self.mark_net(pre_out)
        if self.use_embedding_weights:
            mark_net_out = F.linear(mark_net_out, self.channel_embedding.weight)  # No bias by default
        
        if self.factored_heads:
            log_mark_probs = F.log_softmax(mark_net_out, dim=-1)
            log_intensity = self.intensity_net(pre_out)
            all_log_mark_intensities = log_mark_probs + log_intensity
            total_intensity = log_intensity.exp().squeeze(dim=-1)
        else:
            mark_net_out = self.softplus(mark_net_out)
            all_log_mark_intensities = torch.log(mark_net_out + 1e-12)
            if self.zero_inflated:
                zero_probs = self.zero_prob_net(mark_net_out)
                all_log_mark_intensities += zero_probs.log()
            total_intensity = mark_net_out.sum(dim=-1)

        return {
            "all_log_mark_intensities": all_log_mark_intensities,
            "total_intensity": total_intensity,
            "zero_probs": zero_probs
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
        use_embedding_weights=True,
    ):
        super().__init__()
        if latent_size is None:
            latent_size = 0
        self.channel_embedding = channel_embedding
        self.time_embedding = time_embedding
        self.channel_embedding_size, self.time_embedding_dim = self.channel_embedding.weight.shape[-1], self.time_embedding.embedding_dim
        
        self.intensity_net = IntensityNet(
            channel_embedding=self.channel_embedding,
            input_size=latent_size + self.time_embedding_dim + recurrent_hidden_size,
            hidden_size=intensity_hidden_size,
            num_layers=num_intensity_layers,
            act_func=act_func,
            dropout=dropout,
            factored_heads=factored_heads,
            use_embedding_weights=use_embedding_weights,
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
        self.estimate_init_state = (latent_size != 0) and estimate_init_state
        if self.estimate_init_state:
            print("ESTIMATING INITIAL STATE")
            self.init_hidden_state_network = nn.Sequential(
                nn.Linear(latent_size, num_recurrent_layers * recurrent_hidden_size),
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
            init_hidden_state = torch.transpose(init_hidden_state, 0, 1).contiguous()
        else:
            init_hidden_state = self.init_hidden_state.expand(-1, recurrent_input.shape[0], -1).contiguous()  # Match batch size
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

        padded_state_values = state_values 

        selected_hidden_states = padded_state_values.gather(dim=1, index=closest_dict["closest_indices"].unsqueeze(-1).expand(-1, -1, padded_state_values.shape[-1]))

        time_embedding = self.time_embedding(timestamps, state_times)

        components = [time_embedding, selected_hidden_states]
        if latent_state is not None:
            components.append(latent_state.unsqueeze(1).expand(latent_state.shape[0], timestamps.shape[1], latent_state.shape[1]))

        return self.intensity_net(*components)

class HawkesDecoder(nn.Module):
    """Decoder module that transforms a set of marks, timestamps, and latent vector into intensity values for different channels."""

    def __init__(
        self,
        channel_embedding,
        time_embedding,
        recurrent_hidden_size,
        latent_size=None,
        estimate_init_state=True,
        zero_inflated=False,
    ):
        super().__init__()
        if latent_size is None:
            latent_size = 0
        self.channel_embedding = channel_embedding
        self.time_embedding = time_embedding
        self.num_channels, self.channel_embedding_size = self.channel_embedding.weight.shape
        self.latent_size = latent_size

        self.recurrent_input_size = self.channel_embedding_size + recurrent_hidden_size + latent_size
        self.recurrent_hidden_size = recurrent_hidden_size
        self.cell_param_network = nn.Linear(self.recurrent_input_size, 7*recurrent_hidden_size, bias=True)
        self.soft_plus_params = torch.nn.Parameter(torch.randn(self.num_channels,) * 0.0001)

        self.hidden_to_intensity_logits = nn.Linear(recurrent_hidden_size, self.num_channels)

        self.estimate_init_state = estimate_init_state
        if self.estimate_init_state:
            print("ESTIMATING INITIAL STATE")
            self.init_hidden_state_network = nn.Sequential(
                nn.Linear(latent_size, 6*recurrent_hidden_size),
            )
        else:
            print("NOT ESTIMATING INITIAL STATE")
            self.register_parameter(
                name="init_hidden_state",
                param=nn.Parameter(xavier_truncated_normal(size=(1, 6*recurrent_hidden_size), no_average=True))
            )
        
        self.zero_inflated = zero_inflated
        if zero_inflated:
            print("ZERO INFLATED NEURAL HAWKES PROCESS")
            self.zero_prob_net = nn.Sequential(nn.Linear(recurrent_hidden_size, self.num_channels), nn.Sigmoid())

    def get_init_states(self, batch_size, latent_state=None):
        if self.estimate_init_state:
            init_states = self.init_hidden_state_network(latent_state)
        else:
            init_states = self.init_hidden_state.expand(batch_size, -1)
        h_d, c_d, c_bar, c, delta, o = torch.chunk(init_states, 6, -1)
        
        return torch.tanh(h_d), torch.tanh(c_d), torch.tanh(c_bar), torch.tanh(c), F.softplus(delta), torch.sigmoid(o)

    def recurrence(self, emb_event_t, h_d_tm1, c_tm1, c_bar_tm1):
        feed = torch.cat((emb_event_t, h_d_tm1), dim=1)
        # B * 2H
        (gate_i,
        gate_f,
        gate_z,
        gate_o,
        gate_i_bar,
        gate_f_bar,
        gate_delta) = torch.chunk(self.cell_param_network(feed), 7, -1)

        gate_i = torch.sigmoid(gate_i)
        gate_f = torch.sigmoid(gate_f)
        gate_z = torch.tanh(gate_z)
        gate_o = torch.sigmoid(gate_o)
        gate_i_bar = torch.sigmoid(gate_i_bar)
        gate_f_bar = torch.sigmoid(gate_f_bar)
        gate_delta = F.softplus(gate_delta)

        c_t = gate_f * c_tm1 + gate_i * gate_z
        c_bar_t = gate_f_bar * c_bar_tm1 + gate_i_bar * gate_z

        return c_t, c_bar_t, gate_o, gate_delta

    def decay(self, c_t, c_bar_t, o_t, delta_t, duration_t):
        c_d_t = c_bar_t + (c_t - c_bar_t) * \
            torch.exp(-delta_t * duration_t)

        h_d_t = o_t * torch.tanh(c_d_t)

        return c_d_t, h_d_t

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

        time_deltas = self.time_embedding(timestamps)
        
        components = []
        components.append(self.channel_embedding(marks))

        if latent_state is not None:
             components.append(latent_state.unsqueeze(1).expand(latent_state.shape[0], timestamps.shape[1], latent_state.shape[1]))

        recurrent_input = torch.cat(components, dim=-1)
        assert(recurrent_input.shape[-1] == (self.recurrent_input_size - self.recurrent_hidden_size))
        
        h_d, c_d, c_bar, c, delta_t, o_t = self.get_init_states(time_deltas.shape[0], latent_state)
        hidden_states = [torch.cat((h_d, o_t, c_bar, c, delta_t), -1)]
        for i in range(time_deltas.shape[1]):
            r_input, t_input = recurrent_input[:, i, :], time_deltas[:, i, :]
            c, c_bar, o_t, delta_t = self.recurrence(r_input, h_d, c_d, c_bar)
            c_d, h_d = self.decay(c, c_bar, o_t, delta_t, t_input)
            hidden_states.append(torch.cat((h_d, o_t, c_bar, c, delta_t), -1))
            
        hidden_states = torch.stack(hidden_states, dim=1)
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
        zero_probs = None

        padded_state_values = state_values 

        selected_hidden_states = padded_state_values.gather(dim=1, index=closest_dict["closest_indices"].unsqueeze(-1).expand(-1, -1, padded_state_values.shape[-1]))
        time_embedding = self.time_embedding(timestamps, state_times)
        h_d, o_t, c_bar, c, delta_t = torch.chunk(selected_hidden_states, 5, -1)

        _, h_t = self.decay(c, c_bar, o_t, delta_t, time_embedding)

        intensity_values = F.softplus(self.hidden_to_intensity_logits(h_t))
        all_log_mark_intensities = torch.log(intensity_values + 1e-12)
        if self.zero_inflated:
            zero_probs = self.zero_prob_net(h_t)
            all_log_mark_intensities += torch.log(zero_probs + 1e-12)
        total_intensity = intensity_values.sum(dim=-1)

        return {
            "all_log_mark_intensities": all_log_mark_intensities,
            "total_intensity": total_intensity,
            "zero_probs": zero_probs,
        }

class RMTPPDecoder(nn.Module):
    """Decoder module that transforms a set of marks, timestamps, and latent vector into intensity values for different channels."""

    def __init__(
        self,
        channel_embedding,
        time_embedding,
        recurrent_hidden_size,
        latent_size=None,
        estimate_init_state=True,
        zero_inflated=False,
    ):
        super().__init__()
        if latent_size is None:
            latent_size = 0
        self.channel_embedding = channel_embedding
        self.time_embedding = time_embedding
        self.num_channels, self.channel_embedding_size = self.channel_embedding.weight.shape
        self.latent_size = latent_size

        self.recurrent_input_size = self.channel_embedding_size + latent_size + self.time_embedding.embedding_dim
        self.recurrent_hidden_size = recurrent_hidden_size
        self.recurrent_net = nn.LSTM(
            input_size=self.recurrent_input_size, 
            hidden_size=self.recurrent_hidden_size,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )

        self.hidden_to_intensity_logits = nn.Linear(recurrent_hidden_size, self.num_channels)
        self.time_to_intensity_logits = nn.Linear(1, self.num_channels)
        self.time_to_intensity_logits.weight.data = torch.rand_like(self.time_to_intensity_logits.weight.data)*0.001
        self.time_to_intensity_logits.bias.data = torch.rand_like(self.time_to_intensity_logits.bias.data)*0.001

        self.estimate_init_state = estimate_init_state
        if self.estimate_init_state:
            print("ESTIMATING INITIAL STATE")
            self.init_hidden_state_network = nn.Sequential(
                nn.Linear(latent_size, 2 * recurrent_hidden_size),
            )
        else:
            print("NOT ESTIMATING INITIAL STATE")
            self.register_parameter(
                name="init_hidden_state",
                param=nn.Parameter(xavier_truncated_normal(size=(1, 1, 2*recurrent_hidden_size), no_average=True))
            )

        if zero_inflated:
            print("ZERO INFLATED RMTPP")
            self.zero_prob_net = nn.Sequential(nn.Linear(recurrent_hidden_size, self.num_channels), nn.Sigmoid())


    def get_init_states(self, batch_size, latent_state=None):
        if self.estimate_init_state:
            init_states = self.init_hidden_state_network(latent_state).unsqueeze(0)
        else:
            init_states = self.init_hidden_state.expand(1, batch_size, -1)
        h_0, c_0 = torch.chunk(init_states, 2, -1)
        
        return torch.tanh(h_0), torch.tanh(c_0)

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

        time_deltas = self.time_embedding(timestamps)
        
        components = []
        components.append(self.channel_embedding(marks))
        components.append(time_deltas)

        if latent_state is not None:
             components.append(latent_state.unsqueeze(1).expand(latent_state.shape[0], timestamps.shape[1], latent_state.shape[1]))

        recurrent_input = torch.cat(components, dim=-1)
        assert(recurrent_input.shape[-1] == (self.recurrent_input_size))
        
        init_state = self.get_init_states(time_deltas.shape[0], latent_state)

        hidden_states = [init_state[0].squeeze(0).unsqueeze(1)]
        output_hidden_states, (ohs, ocs) = self.recurrent_net(recurrent_input, init_state)
        hidden_states.append(output_hidden_states)
            
        hidden_states = torch.cat(hidden_states, dim=1)
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
        zero_probs = None

        padded_state_values = state_values 

        selected_hidden_states = padded_state_values.gather(dim=1, index=closest_dict["closest_indices"].unsqueeze(-1).expand(-1, -1, padded_state_values.shape[-1]))
        time_embedding = self.time_embedding(timestamps, state_times)

        hs_logits = self.hidden_to_intensity_logits(selected_hidden_states)
        time_logits = self.time_to_intensity_logits(time_embedding)

        all_log_mark_intensities = hs_logits + time_logits

        if self.zero_inflated:
            zero_probs = self.zero_prob_net(h_t)
            all_log_mark_intensities += torch.log(zero_probs + 1e-12)

        intensity_values = torch.exp(all_log_mark_intensities)
        total_intensity = intensity_values.sum(dim=-1)

        return {
            "all_log_mark_intensities": all_log_mark_intensities,
            "total_intensity": total_intensity,
            "zero_probs": zeor_probs,
        }
