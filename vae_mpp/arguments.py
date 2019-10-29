import argparse
import json

from .utils import print_log


def general_args(parser):
    group = parser.add_argument_group("General set arguments for miscelaneous utilities.")
    #group.add_argument("--json_config_path", default=None, help="Path to json file containing arguments to be parsed.")
    #group.add_argument("--", default=, help="")

def model_config_args(parser):
    group = parser.add_argument_group("Model configuration arguments.")
    #group.add_argument("--", default=, help="")

def training_args(parser):
    group = parser.add_argument_group("Training specification arguments.")
    group.add_argument("--checkpoint_path", default="./", help="")
    group.add_argument("--train_epochs", default=40, help="Number of epochs to iterate over for training.")
    group.add_argument("--batch_size", default=32, help="Number of samples per batch.")
    group.add_argument("--save_epochs", default=1, help="Number of training epochs to complete between model checkpoint saves.")
    group.add_argument("--optimizer", default="adam", help="Type of optimization algorithm to use.")
    group.add_argument("--lr", default=0.0003, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.01, help="L2 coefficient for weight decay.")
    group.add_argument("--warmup_pct", default=0.01, help="Percentage of 'train_iters' to be spent ramping learning rate up from 0.")
    group.add_argument("--lr_decay_style", default="cosine", help="Decay style for the learning rate, after the warmup period.")

def evaluation_args(parser):
    group = parser.add_argument_group("Evaluation specification arguments.")
    #group.add_argument("--", default=, help="")

def sampling_args(parser):
    group = parser.add_argument_group("Sampling specification arguments.")
    #group.add_argument("--", default=, help="")

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
