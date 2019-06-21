"""
Command-line utility for visualizing intensities of an already trained model
"""
import logging
import os
from pathlib import Path
import sys
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from vae_mpp.data import PointPatternDataset, pad_and_combine_instances
from vae_mpp.models import Model
from vae_mpp.parametric_pp import PointProcessFactory

logger = logging.getLogger(__name__)

def _visualize(args):
    '''Training function'''

    config_file_path = args.model_dir / "config.yaml"
    checkpoint_path = args.model_dir / "model.pt"
    vis_output_dir = args.model_dir / "visualizations"

    if not config_file_path.exists():
        logger.error('File config.yaml does not exist. Exiting.')
        sys.exit(1)
    if not checkpoint_path.exists():
        logger.error('File model.pt does not exist. Exiting.')
        sys.exit(1)

    with open(config_file_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    if not vis_output_dir.exists():
        logger.info('Creating directory "{}"'.format(vis_output_dir))
        vis_output_dir.mkdir()

    # Set up logging
    fh = logging.FileHandler(args.model_dir / 'output.log')
    logging.getLogger().addHandler(fh)

    torch.manual_seed(config.get('seed', 5150))
    np.random.seed(config.get('seed', 1336) + 1)

    # Initialize model components from config
    model = Model.from_config(config['model'])

    # Restore checkpoint
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict['model'])

    train_config = config['training']

    eval_dataset = PointPatternDataset(config['evaluation_data'])
    if "point_process_objects" in config:
        with open(config["point_process_objects"], "rb") as f:
            pp_obj_dicts = pickle.load(f)
        pp_objs = [PointProcessFactory(d) for d in pp_obj_dicts]
    else:
        pp_objs = None

    data_loader = DataLoader(eval_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=train_config.get('num_workers', 0),
                              collate_fn=pad_and_combine_instances)
    data_tqdm = tqdm(data_loader, desc='')
    pp_obj_idx = 0
    colors = sns.color_palette()
    for instance in data_tqdm:
        # Process Batch
        instance["mc_samples"] = 2000
        instance["T"] = train_config.get("train_right_window", None)
        with torch.no_grad():
            output = model(**instance)

        fig, ax = plt.subplots(1, 1, figsize=(12,8))
        all_times = np.array(output["all_times"].squeeze())
        model_intensities = torch.exp(output["log_intensities"] + output["log_probs"]).squeeze()
        model_intensities = np.array(model_intensities)

        print(model_intensities.shape)
        print(model_intensities)
        for k, v in instance.items():
            try:
                print(k, "-", v.shape)
            except:
                try:
                    print(k, "-", len(v))
                except:
                    print(k)

        for k, v in output.items():
            try:
                print(k, "-", v.shape)
            except:
                try:
                    print(k, "-", len(v))
                except:
                    print(k)

        sys.exit()
        for k in range(model_intensities.shape[1]):
            ax.plot(all_times, model_intensities[:, k], color=colors[k], label="Model - k={}".format(k))

        actual_times = instance["times"].squeeze().tolist()
        actual_marks = instance["marks"].squeeze().tolist()

        if pp_objs is not None:
            pp_obj = pp_objs[pp_obj_idx]
            pp_obj.clear()
            for t,m in zip(actual_times, actual_marks):
                pp_obj.update(t, m, 0)
            actual_intensities = pp_obj.intensity(all_times)

            for k in range(model_intensities.shape[1]):
                ax.plot(all_times, actual_intensities[:, k], color=colors[k], alpha=0.5, linestyle='dashed', label="Real  - k={}".format(k))

        for pt, mark in zip(actual_times, actual_marks):
            ax.axvline(pt, color=colors[mark], alpha=0.2, linestyle='dotted')

        ax.set_ylabel("Intensity by Mark")
        ax.set_xlabel("Time")
        ax.legend()
        fig.savefig(vis_output_dir / "example_{}.png".format(pp_obj_idx),
                    dpi=400,
                    bbox_inches="tight")

        pp_obj_idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('model_dir', type=Path, help='directory containing config file and model checkpoint file')
    args, _ = parser.parse_known_args()

    if os.environ.get("DEBUG"):
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=level)

    _visualize(args)