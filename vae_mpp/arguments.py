import argparse
from utils import print_log
import json


def general_args(parser):
    group = parser.add_argument_group("General set arguments for miscelaneous utilities.")
    group.add_argument("--json_config_path", default=None, help="Path to json file containing arguments to be parsed.")
    #group.add_argument("--", default=, help="")

def model_config_args(parser):
    group = parser.add_argument_group("Model configuration arguments.")
    #group.add_argument("--", default=, help="")

def training_args(parser):
    group = parser.add_argument_group("Training specification arguments.")
    #group.add_argument("--", default=, help="")

def evaluation_args(parser):
    group = parser.add_argument_group("Evaluation specification arguments.")
    #group.add_argument("--", default=, help="")

def sampling_args(parser):
    group = parser.add_argument_group("Sampling specification arguments.")
    #group.add_argument("--", default=, help="")

def load_args(args):
    '''Override key/value pairs in args dictionary with those in the json config file.'''
    args.update(json.load(open(args.json_config_path, "r")))

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

    if args.json_config_path:
        load_args(args)

    if args.print_args:
        print_args(args)
