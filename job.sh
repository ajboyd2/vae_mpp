python -m vae_mpp.train \
    --seed 123 \
    --cuda \
    --time_embedding_size 8 \
    --use_raw_time \
    --channel_embedding_size 3 \
    --num_channels 3 \
    --enc_hidden_size 8 \
    --enc_num_recurrent_layers 2 \
    --latent_size 3 \
    --agg_method "concat" \
    --agg_noise \
    --use_encoder \
    --dec_recurrent_hidden_size 3 \
    --dec_num_recurrent_layers 1 \
    --dec_intensity_hidden_size 8 \
    --dec_num_intensity_layers 2 \
    --dec_act_func "relu" \
    --dropout 0.2 \
    --checkpoint_path "/home/alexjb/hpp2_results/" \
    --train_epochs 10 \
    --train_data_path "./data/hpp2/train/" \
    --num_workers 2 \
    --batch_size 32 \
    --log_interval 100 \
    --save_epochs 1 \
    --optimizer "adam" \
    --grad_clip 10000.0 \
    --lr 0.001 \
    --loss_beta 1.0 \
    --loss_lambda 0.0 \
    --weight_decay 0.0 \
    --warmup_pct 0.001 \
    --lr_decay_style "constant" \
    --valid_data_path "./data/hpp2/valid/" \
    --loss_cyclical 0.1 \
#    --not_amortized  
#    --dec_intensity_use_embeddings \
#    --dec_intensity_factored_heads \

python -m vae_mpp.evaluate \
    --seed 123 \
    --cuda \
    --time_embedding_size 8 \
    --use_raw_time \
    --channel_embedding_size 3 \
    --num_channels 3 \
    --enc_hidden_size 8 \
    --enc_num_recurrent_layers 2 \
    --latent_size 3 \
    --agg_method "concat" \
    --agg_noise \
    --use_encoder \
    --dec_recurrent_hidden_size 3 \
    --dec_num_recurrent_layers 1 \
    --dec_intensity_hidden_size 8 \
    --dec_num_intensity_layers 2 \
    --dec_act_func "relu" \
    --dropout 0.2 \
    --checkpoint_path "/home/alexjb/hpp2_results/" \
    --train_epochs 5 \
    --train_data_path "./data/hpp2/vis/" \
    --num_workers 2 \
    --batch_size 32 \
    --log_interval 100 \
    --save_epochs 1 \
    --optimizer "adam" \
    --grad_clip 1.0 \
    --lr 0.01 \
    --loss_beta 1.0 \
    --loss_lambda 0.0 \
    --weight_decay 0.01 \
    --warmup_pct 0.01 \
    --lr_decay_style "cosine" \
    --valid_data_path "./data/hpp2/vis/"  \
#    --not_amortized  \
#    --dec_intensity_factored_heads \
#    --dec_intensity_use_embeddings \
