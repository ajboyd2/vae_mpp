python3 -m vae_mpp.train \
    --seed 123 \
    --time_embedding_size 32 \
    --use_raw_time \
    --use_delta_time \
    --channel_embedding_size 32 \
    --num_channels 3 \
    --enc_hidden_size 16 \
    --enc_bidirectional \
    --enc_num_recurrent_layers 2 \
    --agg_method "concat" \
    --agg_noise \
    --dec_recurrent_hidden_size 32 \
    --dec_num_recurrent_layers 2 \
    --dec_intensity_hidden_size 16 \
    --dec_num_intensity_layers 1 \
    --dec_act_func "gelu" \
    --dropout 0.2 \
    --checkpoint_path "./" \
    --train_epochs 40 \
    --train_data_path "./data/1_pp/training.pickle" \
    --num_workers 0 \
    --batch_size 32 \
    --log_interval 100 \
    --save_epochs 1 \
    --optimizer "adam" \
    --grad_clip 1.0 \
    --lr 0.0001 \
    --loss_alpha 0.1 \
    --loss_lambda 100.0 \
    --weight_decay 0.01 \
    --warmup_pct 0.01 \
    --lr_decay_style "cosine" \
    --valid_data_path "./data/1_pp/validation.pickle" \
    --use_encoder \
#    --dont_print_args \
