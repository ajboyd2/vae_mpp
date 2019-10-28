from vae_mpp.modules import PPDecoder, PPEncoder,PPAggregator, TemporalEmbedding
from vae_mpp.models.model import PPModel

def make_model(
    enc_hidden_size,
    dec_hidden_size,
    
    dec_act_func="gelu",
    dropout=0.2,
):


enc
channel_embedding, time_embedding, hidden_size, bidirectional, num_recurrent_layers, dropout,

agg
self, method, hidden_size, noise=True

dec
channel_embedding, time_embedding, act_func, num_intensity_layers, intensity_hidden_size, num_recurrent_layers, recurrent_hidden_size, dropout, latent_size=None,

