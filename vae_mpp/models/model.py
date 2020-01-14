import torch
import torch.nn as nn
import torch.nn.functional as F

NORMS = (
    nn.LayerNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LocalResponseNorm,
    #nn.SyncBatchNorm,
)

class PPModel(nn.Module):

    def __init__(
        self, 
        decoder, 
        encoder=None, 
        aggregator=None, 
        amortized=True,
        p_z=torch.distributions.Laplace,
        q_z_x=torch.distributions.Laplace,
    ):
        """Constructor for general PPModel class.
        
        Arguments:
            decoder {torch.nn.Module} -- Neural network decoder that accepts a latent state, marks, timestamps, and times of sample points. 

        Keyword Arguments:
            time_embedding {torch.nn.Module} -- Function to transform k-dimensional timestamps into (k+1)-dimensional embedded vectors. If specified, will make encoder and decoder share this function. (default: {None})
            encoder {torch.nn.Module} -- Neural network encoder that accepts marks and timestamps and returns a single latent state (default: {None})
            aggregator {torch.nn.Module} -- Module that turns a tensor of hidden states into a latent vector (with noise added during training) (default: {None})
        """
        super().__init__()

        self.decoder = decoder
        self.encoder = encoder
        self.aggregator = aggregator
        self.has_encoder = (encoder is not None) and (aggregator is not None)
        self.amortized = amortized
        if not amortized:
            self.latent_mu = nn.Embedding(num_embeddings=100, embedding_dim=decoder.latent_size)
            #self.latent_log_var = nn.Embedding(num_embeddings=100, embedding_dim=decoder.latent_size)
            self.latent_sigma = nn.Embedding(num_embeddings=100, embedding_dim=decoder.latent_size)
        
        self.p_z = p_z
        self.q_z_x = q_z_x
        self._p_z_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, decoder.latent_size), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, decoder.latent_size), requires_grad=False)  # logvar
        ])

    def get_prior_params(self):
        return self._p_z_params[0], F.softmax(self._p_z_params[1], dim=-1) * self._p_z_params[1].size(-1)

    def get_states(self, tgt_marks, tgt_timestamps, latent_state):
        """Get the hidden states that can be used to extract intensity values from."""
        
        states = self.decoder.get_states(
            marks=tgt_marks, 
            timestamps=tgt_timestamps, 
            latent_state=latent_state,
        )

        return {
            "state_values": states,
            "state_times": tgt_timestamps,
        }

    def get_intensity(self, state_values, state_times, timestamps, latent_state, marks=None):
        """Given a set of hidden states, timestamps, and latent_state get a tensor representing intensity values at timestamps.
        Specify marks to get intensity values for specific channels."""

        intensity_dict = self.decoder.get_intensity(
            state_values=state_values,
            state_times=state_times,
            timestamps=timestamps,
            latent_state=latent_state,
        )

        if marks is not None:
            #gathered_probs = intensity_dict["log_mark_probs"].gather(dim=-1, index=marks.unsqueeze(-1)).squeeze(-1)
            #intensity_dict["log_mark_intensity"] = gathered_probs + intensity_dict["log_intensity"]
            intensity_dict["log_mark_intensity"] = intensity_dict["all_log_mark_intensities"].gather(dim=-1, index=marks.unsqueeze(-1)).squeeze(-1)
        
        return intensity_dict 
        
    def get_latent(self, ref_marks_fwd, ref_timestamps_fwd, ref_marks_bwd, ref_timestamps_bwd, context_lengths, pp_id):
        """Computes latent variable for a given set of reference marks and timestamped events."""
        if self.amortized:
            hidden_states = self.encoder(
                forward_marks=ref_marks_fwd,
                forward_timestamps=ref_timestamps_fwd,
                backward_marks=ref_marks_bwd,
                backward_timestamps=ref_timestamps_bwd,
            )

            return self.aggregator(hidden_states, context_lengths)
        else:
            #mu, log_var = self.latent_mu(pp_id), self.latent_log_var(pp_id)
            #mu, log_var = mu.squeeze(dim=1), log_var.squeeze(dim=1)
            raise NotImplementedError
            mu, sigma = self.latent_mu(pp_id), self.latent_sigma(pp_id)
            mu, sigma = mu.squeeze(dim=1), sigma.squeeze(dim=1)

            if self.training:
                #latent_state = torch.randn_like(mu) * torch.exp(log_var / 2.0) + mu
                latent_state = torch.randn_like(mu) * sigma + mu
            else:
                latent_state = mu

            return {
                "latent_state": latent_state,
                "mu": mu,
                "sigma": sigma,
            }

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
        if self.has_encoder:
            latent_state_dict = self.get_latent(
                ref_marks_fwd=ref_marks,
                ref_timestamps_fwd=ref_timestamps,
                ref_marks_bwd=ref_marks_bwd,
                ref_timestamps_bwd=ref_timestamps_bwd,
                context_lengths=context_lengths,
                pp_id=pp_id,
            )
        else:
            latent_state_dict = {
                "latent_state": None,
                "q_z_x": None,
            }
        latent_state = latent_state_dict["latent_state"]
        return_dict["latent_state"] = latent_state
        return_dict["q_z_x"] = latent_state_dict["q_z_x"]
        return_dict["p_z"] = self.p_z(*self.get_prior_params())        

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

    @staticmethod
    def log_likelihood(return_dict, right_window, left_window=0.0, mask=None):
        """Computes per-batch log-likelihood from the results of a forward pass (that included a set of sample points). 
        
        Arguments:
            return_dict {dict} -- Output from a forward call where `tgt_marks` and `sample_timestamps` were not None
            right_window {float} -- Upper-most value that was considered when the sampled points were generated

        Keyword Arguments:
            left_window {float} -- Lower-most value that was considered when the sampled points were generated (default: {0})
            mask {FloatTensor} -- Mask to delineate target intensities that correspond to real events and paddings (default: {None}) 
        """

        assert("tgt_intensities" in return_dict and "log_mark_intensity" in return_dict["tgt_intensities"])
        assert("sample_intensities" in return_dict)

        if mask is None:
            mask = 1
        else:
            assert(all(x == y for x,y in zip(return_dict["tgt_intensities"]["log_mark_intensity"].shape, mask.shape)))  # make sure they are same size

        # num_samples = return_dict["sample_intensities"]["log_intensity"].shape[1]
        log_mark_intensity = return_dict["tgt_intensities"]["log_mark_intensity"]
        positive_samples = torch.where(mask, log_mark_intensity, torch.zeros_like(log_mark_intensity)).sum(dim=-1)
        negative_samples = (right_window - left_window) * return_dict["sample_intensities"]["total_intensity"].mean(dim=-1)  # Summing and divided by number of samples

        return {
            "log_likelihood": ((1.0 * positive_samples) - negative_samples).mean(),
            "positive_contribution": positive_samples.mean(),
            "negative_contribution": negative_samples.mean(),
        }

            
    def get_param_groups(self):
        """Returns iterable of dictionaries specifying parameter groups.
        The first dictionary in the return value contains parameters that will be subject to weight decay.
        The second dictionary in the return value contains parameters that will not be subject to weight decay.
        
        Returns:
            (param_group, param_groups) -- Tuple containing sets of parameters, one of which has weight decay enabled, one of which has it disabled.
        """

        weight_decay_params = {'params': []}
        no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
        for module_ in self.modules():
            # Doesn't make sense to decay weights for a LayerNorm, BatchNorm, etc.
            if isinstance(module_, NORMS):
                no_weight_decay_params['params'].extend([
                    p for p in module_._parameters.values() if p is not None
                ])
            else:
                # Also doesn't make sense to decay biases.
                weight_decay_params['params'].extend([
                    p for n, p in module_._parameters.items() if p is not None and n != 'bias'
                ])
                no_weight_decay_params['params'].extend([
                    p for n, p in module_._parameters.items() if p is not None and n == 'bias'
                ])

        return weight_decay_params, no_weight_decay_params