import torch
import torch.nn as nn

_activations = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'identity': lambda x: (lambda y: y)(x)
}

class PP_Decoder(nn.Module):

    def __init__(self,
                 num_events,
                 hidden_size,
                 event_embedding_size,
                 time_embedding_size,
                 latent_size,
                 intensity_layer_size,
                 intensity_num_layers,
                 intensity_act_func,
                 layer_norm):
        super(PP_Decoder, self).__init__()

        self.num_events = num_events
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.event_embedding_size = event_embedding_size
        self.time_embedding_size = time_embedding_size

        if layer_norm in _activations:
            self.layer_norm = _activations[layer_norm]
        elif layer_norm == "layer_norm":
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            raise ModuleNotFoundError("'{}' is not a supported layer normalization operation.".format(layer_norm))

        self.register_buffer("hidden0", torch.zeros(1, hidden_size))
        self.register_buffer("latent0", torch.FloatTensor())

        self.rnn_cell = nn.GRUCell(input_size=latent_size + event_embedding_size + time_embedding_size,
                                   hidden_size=hidden_size)
        if intensity_act_func in _activations:
            act_func = _activations[intensity_act_func]
        else:
            raise ModuleNotFoundError("'{}' is not a supported activation function.".format(intensity_act_func))

        intensity_layers = list()
        intensity_layers.append(nn.Linear(latent_size + hidden_size + time_embedding_size, intensity_layer_size))
        intensity_layers.append(act_func())

        for _ in range(intensity_num_layers - 1):
            intensity_layers.append(nn.Linear(intensity_layer_size, intensity_layer_size))
            intensity_layers.append(act_func())

        self.intensity_net_pre = nn.Sequential(*intensity_layers)

        self.intensity_net = nn.Sequential(
            nn.Linear(intensity_layer_size, 1)
        )

        self.mark_net = nn.Sequential(
            nn.Linear(intensity_layer_size, num_events),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, time_embed, mark_embed, mask=None, latent_state=None):
        # time_embed.shape = (seq, batch, embed)
        # mark_embed.shape = (seq, batch, embed)
        # mask.shape       = (seq, batch)
        seq_len, batch_size, _ = time_embed.shape
        hidden_state = self.hidden0.expand(batch_size, -1)
        if latent_state is None:
            latent_state = self.latent0

        output = {
            "log_intensities": [],
            "log_probs": [],
            "pre_out": [],
            "hidden_states": [],
            "mask": mask
        }

        for i in range(seq_len):
            time_step = time_embed[i, :, :]
            mark_step = mark_embed[i, :, :]

            if mask is not None:
                mask_step = mask[i, :].unsqueeze(-1).expand(-1, self.hidden_size)

            pre_out = self.intensity_net_pre(
                torch.cat((latent_state, hidden_state, time_step), dim=-1)
            )

            output["log_intensities"].append(self.intensity_net(pre_out))
            output["log_probs"].append(self.mark_net(pre_out))
            output["pre_out"].append(pre_out)

            hidden_state_prime = self.layer_norm(self.rnn_cell(
                torch.cat((latent_state, time_step, mark_step), dim=-1),
                hidden_state
            ))

            if mask is not None:
                hidden_state = torch.where(mask_step, hidden_state_prime, hidden_state)
            else:
                hidden_state = hidden_state_prime

            output["hidden_states"].append(hidden_state)

        output["log_intensities"] = torch.stack(output["log_intensities"], dim=0)
        output["log_probs"] = torch.stack(output["log_probs"], dim=0)
        return output
