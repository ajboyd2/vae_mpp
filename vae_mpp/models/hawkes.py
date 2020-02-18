import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import PPModel
from vae_mpp.modules.utils import xavier_truncated_normal, flatten, find_closest, ACTIVATIONS

class HawkesModel(PPModel):

    def __init__(
        self, 
        num_marks,
        bounded=False,
    ):
        """Constructor for general PPModel class.
        
        Arguments:
            decoder {torch.nn.Module} -- Neural network decoder that accepts a latent state, marks, timestamps, and times of sample points. 

        Keyword Arguments:
            time_embedding {torch.nn.Module} -- Function to transform k-dimensional timestamps into (k+1)-dimensional embedded vectors. If specified, will make encoder and decoder share this function. (default: {None})
            encoder {torch.nn.Module} -- Neural network encoder that accepts marks and timestamps and returns a single latent state (default: {None})
            aggregator {torch.nn.Module} -- Module that turns a tensor of hidden states into a latent vector (with noise added during training) (default: {None})
        """
        super().__init__(decoder=None)

        self.num_marks = num_marks
        self.alphas = torch.nn.Embedding(
            num_embeddings=num_marks, 
            embedding_dim=num_marks,
        )
        self.deltas = torch.nn.Embedding(
            num_embeddings=num_marks,
            embedding_dim=num_marks,
        )
        self.alphas.weight.data = torch.randn_like(self.alphas.weight.data) * 0.0001
        self.deltas.weight.data = torch.randn_like(self.deltas.weight.data) * 0.0001
        
        self.mus = torch.nn.Parameter(torch.randn(num_marks,) * 0.0001)
        self.s = torch.nn.Parameter(torch.randn(num_marks,) * 0.0001)
        self.bounded = bounded

    def get_states(self, tgt_marks, tgt_timestamps, latent_state):
        """Get the hidden states that can be used to extract intensity values from."""

        return {
            "state_values": tgt_marks,
            "state_times": tgt_timestamps,
        }

    def get_intensity(self, state_values, state_times, timestamps, latent_state, marks=None):
        """Given a set of hidden states, timestamps, and latent_state get a tensor representing intensity values at timestamps.
        Specify marks to get intensity values for specific channels."""

        batch_size, seq_len = timestamps.shape
        hist_len = state_times.shape[1]
        num_marks = self.num_marks

        mu, alpha, delta = self.mus, self.alphas(state_values), self.deltas(state_values)
        if self.bounded:
            mu, alpha, delta = mu.exp(), alpha.exp(), delta.exp()
        mu = mu.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        alpha = torch.transpose(alpha.unsqueeze(-1).expand(-1, -1, -1, seq_len), 1, 3).contiguous()
        delta = torch.transpose(delta.unsqueeze(-1).expand(-1, -1, -1, seq_len), 1, 3).contiguous()

        time_diffs = F.relu(timestamps.unsqueeze(2) - state_times.unsqueeze(1))
        time_diffs = time_diffs.unsqueeze(2).expand(-1, -1, num_marks, -1)
        valid_terms = time_diffs > 0

        prod = alpha * (-1 * delta * time_diffs).exp()
        prod = torch.where(valid_terms, prod, torch.zeros_like(prod))

        all_mark_intensities = mu + prod.sum(-1)

        if not self.bounded:
            s = self.s.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1).exp()
            all_mark_intensities = s * torch.log(1 + torch.exp(all_mark_intensities / s))

        all_log_mark_intensities = all_mark_intensities.log()
        total_intensity = all_mark_intensities.sum(-1)

        intensity_dict = {
            "all_log_mark_intensities": all_log_mark_intensities,
            "total_intensity": total_intensity,
        }

        if marks is not None:
            intensity_dict["log_mark_intensity"] = intensity_dict["all_log_mark_intensities"].gather(dim=-1, index=marks.unsqueeze(-1)).squeeze(-1)
        
        return intensity_dict 
        
    def forward(self, ref_marks, ref_timestamps, ref_marks_bwd, ref_timestamps_bwd, tgt_marks, tgt_timestamps, context_lengths, sample_timestamps=None, pp_id=None):
        """Encodes a(n optional) set of marks and timestamps into a latent vector, 
        then decodes corresponding intensity values for a target set of timestamps and marks 
        (as well as a sample set if specified).
        
        Arguments:
            ref_marks {torch.LongTensor} -- Tensor containing mark ids that correspond to channel embeddings. Part of the reference set to be encoded.
            ref_timestamps {torch.FloatTensor} -- Tensor containing times that correspond to the events in `ref_marks`. Part of the reference set to be encoded.
            ref_marks_bwd {torch.LongTensor} -- Tensor containing reverse mark ids that correspond to channel embeddings. Part of the reference set to be encoded.
            ref_timestamps_bwd {torch.FloatTensor} -- Tensor containing reverse times that correspond to the events in `ref_marks`. Part of the reference set to be encoded.
            tgt_marks {torch.FloatTensor} -- Tensor containing mark ids that correspond to channel embeddings. These events will be decoded and are assumed to have happened.
            tgt_timestamps {torch.FloatTensor} -- Tensor containing times that correspond to the events in `tgt_marks`. These times will be decoded and are assumed to have happened.
            context_lengths {torch.LongTensor} -- Tensor containing position ids that correspond to last events in the reference material.

        Keyword Arguments:
            sample_timestamps {torch.FloatTensor} -- Times that will have intensity values generated for. These events are _not_ assumed to have happened. (default: {None})
        
        Returns:
            dict -- Dictionary containing the produced latent vector, intermediate hidden states, and intensity values for target sequence and sample points.
        """
        return_dict = {}

        # Encoding phase
        latent_state_dict = {
            "latent_state": None,
            "q_z_x": None,
        }
        latent_state = latent_state_dict["latent_state"]
        return_dict["latent_state"] = latent_state
        return_dict["q_z_x"] = latent_state_dict["q_z_x"]
        return_dict["p_z"] = None

        # Decoding phase
        intensity_state_dict = self.get_states(
            tgt_marks=tgt_marks,
            tgt_timestamps=tgt_timestamps,
            latent_state=latent_state,
        )
        return_dict["state_dict"] = intensity_state_dict

        tgt_intensities = self.get_intensity(
            state_values=intensity_state_dict["state_values"],
            state_times=intensity_state_dict["state_times"],
            timestamps=tgt_timestamps,
            latent_state=latent_state,
            marks=tgt_marks,
        )
        return_dict["tgt_intensities"] = tgt_intensities

        # Sample intensities for objective function
        if sample_timestamps is not None:

            sample_intensities = self.get_intensity(
                state_values=intensity_state_dict["state_values"],
                state_times=intensity_state_dict["state_times"],
                timestamps=sample_timestamps,
                latent_state=latent_state,
                marks=None,
            )
            return_dict["sample_intensities"] = sample_intensities


        return return_dict