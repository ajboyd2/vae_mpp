from torch import nn

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

    def __init__(self, decoder, time_embedding=None, encoder=None, aggregator=None):
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

        if time_embedding:
            self.decoder.set_time_embedding(time_embedding)
            if self.encoder:
                self.encoder.set_time_embedding(time_embedding)
    
    def get_states(self, tgt_marks, tgt_timestamps, latent_state):
        """Get the hidden states that can be used to extract intensity values from.
        
        Arguments:
            tgt_marks {[type]} -- [description]
            tgt_timestamps {[type]} -- [description]
            latent_state {[type]} -- [description]

        Returns:
            dict -- [description]
        """
        
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
        Specify marks to get intensity values for specific channels.
        
        Arguments:
            states {[type]} -- [description]
            marks {[type]} -- [description]
            timestamps {[type]} -- [description]
            latent_state {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """

        intensities = self.decoder.get_intensity(
            state_values=state_values,
            state_times=state_times,
            timestamps=timestamps,
            latent_state=latent_state,
        )

        if marks:
            return intensities.gather(dim=-1, index=marks.unsqueeze(-1))
        else:
            return intensities 
        
    def get_latent(self, ref_marks, ref_timestamps):
        """Computes latent variable for a given set of reference marks and timestamped events.
        
        Arguments:
            ref_marks {[type]} -- [description]
            ref_timestamps {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        hidden_states = self.encoder(
            marks=ref_marks,
            timestamps=ref_timestamps,
        )

        return self.aggregator(hidden_states)

    def forward(self, ref_marks, ref_timestamps, tgt_timestamps, tgt_marks=None, sample_timestamps=None):
        """Encodes a(n optional) set of marks and timestamps into a latent vector, 
        then decodes corresponding intensity values for a target set of timestamps and marks 
        (as well as a sample set if specified).
        
        Arguments:
            ref_marks {[type]} -- [description]
            ref_timestamps {[type]} -- [description]
            tgt_timestamps {[type]} -- [description]
        
        Keyword Arguments:
            tgt_marks {[type]} -- [description] (default: {None})
            sample_timestamps {[type]} -- [description] (default: {None})
        
        Returns:
            [type] -- [description]
        """
        return_dict = {}

        # Encoding phase
        if self.has_encoder:
            latent_state = self.get_latent(
                ref_marks=ref_marks,
                ref_timestamps=ref_timestamps,
            )
        else:
            latent_state = None
        return_dict["latent_state"] = latent_state

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
        if sample_timestamps:
            sample_intensities = self.get_intensity(
                state_values=intensity_state_dict["state_values"],
                state_times=intensity_state_dict["state_times"],
                timestamps=latent_state,
                latent_state=latent_state,
                marks=None,
            )
            return_dict["sample_intensities"] = sample_intensities
        
        return return_dict
            
    def get_param_groups(self):
        """Returns iterable of dictionaries specifying parameter groups.
        The first dictionary in the return value contains parameters that will be subject to weight decay.
        The second dictionary in the return value contains parameters that will not be subject to weight decay.
        
        Returns:
            (param_group, param_groupS) -- [description]
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