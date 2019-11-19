import torch

from vae_mpp.modules import PPDecoder, PPEncoder,PPAggregator, TemporalEmbedding
from vae_mpp.models.model import PPModel

def get_model(
    time_embedding_size, 
    use_raw_time, 
    use_delta_time, 
    max_period,
    channel_embedding_size,
    num_channels,
    enc_hidden_size,
    enc_bidirectional, 
    enc_num_recurrent_layers,
    latent_size,
    agg_method,
    agg_noise,
    use_encoder,
    dec_recurrent_hidden_size,
    dec_num_recurrent_layers,
    dec_intensity_hidden_size,
    dec_intensity_factored_heads,
    dec_num_intensity_layers,
    dec_intensity_use_embeddings,
    dec_act_func="gelu",
    dropout=0.2,
):

    time_embedding = TemporalEmbedding(
        embedding_dim=time_embedding_size,
        use_raw_time=use_raw_time,
        use_delta_time=use_delta_time,
        learnable_delta_weights=True,
        max_period=max_period,
    )

    channel_embedding = torch.nn.Embedding(
        num_embeddings=num_channels, 
        embedding_dim=channel_embedding_size
    )

    if use_encoder:
        encoder = PPEncoder(
            channel_embedding=channel_embedding,
            time_embedding=time_embedding,
            hidden_size=enc_hidden_size // (2 if enc_bidirectional else 1),  # This produces output hidden sizes of enc_hidden_size no matter what
            bidirectional=enc_bidirectional,
            num_recurrent_layers=enc_num_recurrent_layers,
            dropout=dropout 
        )

        aggregator = PPAggregator(
            method=agg_method,
            hidden_size=enc_hidden_size,
            latent_size=latent_size,
            noise=agg_noise,
        )
    else:
        encoder = None
        aggregator = None

    decoder = PPDecoder(
        channel_embedding=channel_embedding,
        time_embedding=time_embedding,
        act_func=dec_act_func,
        num_intensity_layers=dec_num_intensity_layers,
        intensity_hidden_size=dec_intensity_hidden_size,
        num_recurrent_layers=dec_num_recurrent_layers,
        recurrent_hidden_size=dec_recurrent_hidden_size,
        dropout=dropout,
        latent_size=latent_size if use_encoder else 0,
        factored_heads=dec_intensity_factored_heads,
        use_embedding_weights=dec_intensity_use_embeddings,
    )

    return PPModel(
        decoder=decoder,
        encoder=encoder,
        aggregator=aggregator,
    )
