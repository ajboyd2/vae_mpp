import argparse
import json

from .utils import print_log


def general_args(parser):
    group = parser.add_argument_group("General set arguments for miscelaneous utilities.")
    #group.add_argument("--json_config_path", default=None, help="Path to json file containing arguments to be parsed.")
    group.add_argument("--seed", type=int, default=1234321, help="Seed for all random processes.")

def model_config_args(parser):
    group = parser.add_argument_group("Model configuration arguments.")
    group.add_argument("--time_embedding_size", type=int, default=16, help="Size of temporal embeddings.")
    group.add_argument("--use_raw_time", action="store_true", help="Use raw time for encoding temporal information.")
    group.add_argument("--use_delta_time", action="store_true", help="Use time differences for encoding temporal information.")
    group.add_argument("--channel_embedding_size", type=int, default=16, help="Size of mark embeddings.")
    group.add_argument("--num_channels", type=int, default=3, help="Number of different possible marks.")
    group.add_argument("--enc_hidden_size", type=int, default=8, help="Hidden size for encoder GRU.")
    group.add_argument("--enc_bidirectional", action="store_true", help="Enables bidirectional encoding.")
    group.add_argument("--enc_num_recurrent_layers", type=int, default=1, help="Number of recurrent GRU layers in encoder.")
    group.add_argument("--agg_method", type=str, default="concat", help="Method to use for aggregating hidden states from encoder")
    group.add_argument("--agg_noise", action="store_true", help="Add random noise during training to encoded latent vector.")
    group.add_argument("--use_encoder", action="store_true", help="Setup the model in a VAE fashion.")
    group.add_argument("--dec_recurrent_hidden_size", type=int, default=16, help="Hidden size for decoder GRU.")
    group.add_argument("--dec_num_recurrent_layers", type=int, default=1, help="Number of recurrent layers in decoder.")
    group.add_argument("--dec_intensity_hidden_size", type=int, default=16, help="Hidden size of intermediate layers in intensity network.")
    group.add_argument("--dec_num_intensity_layers", type=int, default=1, help="Number of layers in intensity network.")
    group.add_argument("--dec_act_func", type=str, default="gelu", help="Activation function to be used in intensity network.")
    group.add_argument("--dropout", type=float, default=0.2, help="Dropout rate to be applied to all supported layers during training.")

def training_args(parser):
    group = parser.add_argument_group("Training specification arguments.")
    group.add_argument("--checkpoint_path", type=str, default="./", help="")
    group.add_argument("--train_epochs", type=int, default=40, help="Number of epochs to iterate over for training.")
    group.add_argument("--train_data_path", type=str, default="./", help="Path to training data file.")
    group.add_argument("--num_workers", type=int, default=0, help="Number of parallel workers for data loaders.")
    group.add_argument("--batch_size", type=int, default=32, help="Number of samples per batch.")
    group.add_argument("--log_interval", type=int, default=100, help="Number of batches to complete before printing intermediate results.")
    group.add_argument("--save_epochs", type=int, default=1, help="Number of training epochs to complete between model checkpoint saves.")
    group.add_argument("--optimizer", type=str, default="adam", help="Type of optimization algorithm to use.")
    group.add_argument("--grad_clip", type=float, default=1.0, help="Threshold for gradient clipping.")
    group.add_argument("--lr", type=float, default=0.0003, help="Learning rate.")
    group.add_argument("--loss_alpha", type=float, default=0.2, help="Alpha scaling parameter for loss.")
    group.add_argument("--loss_lambda", type=float, default=2, help="Lambda scaling parameter for loss.")
    group.add_argument("--weight_decay", type=float, default=0.01, help="L2 coefficient for weight decay.")
    group.add_argument("--warmup_pct", type=float, default=0.01, help="Percentage of 'train_iters' to be spent ramping learning rate up from 0.")
    group.add_argument("--lr_decay_style", type=str, default="cosine", help="Decay style for the learning rate, after the warmup period.")

def evaluation_args(parser):
    group = parser.add_argument_group("Evaluation specification arguments.")
    group.add_argument("--valid_data_path", type=str, default="./", help="Path to training data file.")
    #group.add_argument("--", type=, default=, help="")

def sampling_args(parser):
    group = parser.add_argument_group("Sampling specification arguments.")
    #group.add_argument("--", type=, default=, help="")

def print_args(args):
    max_arg_len = max(len(k) for k in args.keys())
    for k, v in args.items():
        print_log("{}{}{}".format(
            k,
            max_arg_len + 3 - len(k),
            v,
        ))

def get_args():
    parser = argparse.ArgumentParser()

    general_args(parser)
    model_config_args(parser)
    training_args(parser)
    evaluation_args(parser)
    sampling_args(parser)

    args = parser.parse_args()

    if args.print_args:
        print_args(args)

    return args
