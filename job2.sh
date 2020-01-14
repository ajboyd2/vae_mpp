python -m vae_mpp.train \
    --seed 123 \
    --cuda \
    --time_embedding_size 64 \
    --use_raw_time \
    --channel_embedding_size 32 \
    --num_channels 3 \
    --enc_hidden_size 64 \
    --enc_num_recurrent_layers 1 \
    --latent_size 16 \
    --agg_method "concat" \
    --agg_noise \
    --use_encoder \
    --dec_recurrent_hidden_size 64 \
    --dec_num_recurrent_layers 1 \
    --dec_intensity_hidden_size 32 \
    --dec_num_intensity_layers 2 \
    --dec_act_func "gelu" \
    --dropout 0.01 \
    --checkpoint_path "/home/alexjb/hpp2_resultsv3/" \
    --train_epochs 50 \
    --train_data_path "./data/hpp2/train/" \
    --num_workers 2 \
    --batch_size 128 \
    --log_interval 100 \
    --save_epochs 1 \
    --optimizer "adam" \
    --grad_clip 10000.0 \
    --lr 0.001 \
    --loss_beta 1.0 \
    --loss_lambda 0.0 \
    --weight_decay 0.0 \
    --warmup_pct 0.01 \
    --lr_decay_style "constant" \
    --valid_data_path "./data/hpp2/valid/" \
    --loss_cyclical 0.1 \
#    --dec_intensity_use_embeddings \
#    --not_amortized  
#    --dec_intensity_factored_heads \

python -m vae_mpp.evaluate \
    --seed 123 \
    --cuda \
    --time_embedding_size 64 \
    --use_raw_time \
    --channel_embedding_size 32 \
    --num_channels 3 \
    --enc_hidden_size 64 \
    --enc_num_recurrent_layers 1 \
    --latent_size 16 \
    --agg_method "concat" \
    --agg_noise \
    --use_encoder \
    --dec_recurrent_hidden_size 64 \
    --dec_num_recurrent_layers 1 \
    --dec_intensity_hidden_size 32 \
    --dec_num_intensity_layers 2 \
    --dec_act_func "gelu" \
    --dropout 0.01 \
    --checkpoint_path "/home/alexjb/hpp2_resultsv3/" \
    --train_epochs 5 \
    --train_data_path "./data/hpp2/vis/" \
    --num_workers 2 \
    --batch_size 128 \
    --log_interval 100 \
    --save_epochs 1 \
    --optimizer "adam" \
    --grad_clip 10000.0 \
    --lr 0.001 \
    --loss_beta 1.0 \
    --loss_lambda 0.0 \
    --weight_decay 0.01 \
    --warmup_pct 0.01 \
    --lr_decay_style "cosine" \
    --valid_data_path "./data/hpp2/vis/"  \
#    --dec_intensity_use_embeddings \
#    --dec_intensity_factored_heads \
#    --not_amortized  \
