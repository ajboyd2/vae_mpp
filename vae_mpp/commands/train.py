"""
Command-line utility for training a model
"""
import logging
import os
from pathlib import Path
import shutil
import sys
import argparse

from apex import amp
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from vae_mpp.data import PointPatternDataset, pad_and_combine_instances
from vae_mpp.models import Model
from vae_mpp.optim import Optimizer, LRScheduler


logger = logging.getLogger(__name__)

def _train(args):
    '''Training function'''

    if not args.config.exists():
        logger.error('Config does not exist. Exiting.')
        sys.exit(1)

    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    if args.output_dir.exists() and not (args.resume or args.force):
        logger.error('Directory "{}" already exists. Exiting.'.format(args.output_dir))
        sys.exit(1)
    else:
        logger.info('Creating directory "{}"'.format(args.output_dir))
        if not args.output_dir.exists():
            args.output_dir.mkdir()
        shutil.copy(args.config, args.output_dir / 'config.yaml')

    # Set up logging
    fh = logging.FileHandler(args.output_dir / 'output.log')
    logging.getLogger().addHandler(fh)

    torch.manual_seed(config.get('seed', 5150))
    np.random.seed(config.get('seed', 1336) + 1)

    # Initialize model components from config
    model = Model.from_config(config['model'])

    optimizer = Optimizer.from_config(config['optimizer'], params=model.parameters())
    if 'lr_scheduler' in config:
        lr_scheduler = LRScheduler.from_config(config['lr_scheduler'])
    else:
        lr_scheduler = None

    if args.cuda:
        logger.info('Using cuda')
        if args.cuda_device is not None:
            model = model.cuda(args.cuda_device)
        else:
            model = model.cuda()

        if args.fp16:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # Restore checkpoint
    checkpoint_path = args.output_dir / 'checkpoint.pt'
    best_checkpoint_path = args.output_dir / 'model.pt'
    if (args.output_dir / 'checkpoint.pt').exists() and args.resume:
        logger.info('Found existing checkpoint. Resuming training.')
        state_dict = torch.load(checkpoint_path)
        start_epoch = state_dict['epoch']
        best_metric = state_dict['best_metric']
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
    else:
        logger.info('No existing checkpoint. Training from scratch.')
        start_epoch = 0
        best_metric = float('inf')

    train_config = config['training']

    train_dataset = PointPatternDataset(config['train_data'])
    validation_dataset = PointPatternDataset(config['validation_data'])

    for epoch in range(start_epoch, train_config['epochs']):
        logger.info('Epoch: %i', epoch)

        # Training loop
        logger.info('Training...')
        model.train()
        train_loader = DataLoader(train_dataset,
                                  batch_size=train_config['batch_size'],
                                  shuffle=True,
                                  num_workers=train_config.get('num_workers', 0),
                                  collate_fn=pad_and_combine_instances)
        train_tqdm = tqdm(train_loader, desc='[T] Loss: NA')
        train_losses = []
        optimizer.zero_grad()

        for instance in train_tqdm:
            if args.cuda:
                instance = {key: value.cuda(args.cuda_device) for key, value in instance.items()}

            # Process Batch
            instance["mc_samples"] = train_config.get("train_mc_samples", 100)
            output_dict = model(**instance)
            loss = output_dict['loss'].mean()

            # update gradients
            if args.fp16:
                with amp.scale_loss(loss,  optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Report Losses
            train_losses.append(loss.item())
            train_tqdm.set_description('[T] Batch Loss: {:08.4f} - Avg Loss: {:08.4f}'.format(loss.item(), sum(train_losses) / len(train_losses)))

        # Validation loop
        logger.info('Validating...')
        model.eval()
        validation_loader = DataLoader(validation_dataset,
                                       batch_size=train_config['batch_size'],
                                       shuffle=False,
                                       num_workers=train_config.get('num_workers', 0),
                                       collate_fn=pad_and_combine_instances)
        validation_tqdm = tqdm(validation_loader, desc='[V] Loss: NA')
        validation_losses = []

        for instance in validation_tqdm:
            if args.cuda:
                instance = {key: value.cuda(args.cuda_device) for key, value in instance.items()}

            with torch.no_grad():
                # Process Batch
                instance["mc_samples"] = train_config.get("valid_mc_samples", 500)
                output_dict = model(**instance)
                loss = output_dict['loss'].mean()

                # Report losses
                validation_losses.append(loss.item())
                validation_tqdm.set_description('[V] Batch Loss: {:08.4f} - Avg Loss: {:08.4f}'.format(loss.item(), sum(validation_losses) / len(validation_losses)))

        # Report losses for the epoch
        metric = sum(validation_losses) / len(validation_losses)
        logger.info('Loss: Train {:08.4f} - Validation {:08.4f}'.format(sum(train_losses) / len(train_losses), metric))

        # Checkpoint
        model_state_dict = model.state_dict()
        state_dict = {
            'model': model_state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_metric': best_metric
        }
        if lr_scheduler:
            state_dict['lr_scheduler'] = lr_scheduler.state_dict()

        # If model is best encountered overwrite best model checkpoint.
        if metric < best_metric:
            logger.info('Best model so far.')
            state_dict['best_metric'] = metric
            best_metric = metric
            torch.save(state_dict, best_checkpoint_path)

        # Save current model.
        torch.save(state_dict, checkpoint_path)

    if lr_scheduler is not None:
        lr_scheduler.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('config', type=Path, help='path to config .yaml file')
    parser.add_argument('output_dir', type=Path, help='output directory to save model to')

    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--cuda-device', type=int, help='CUDA device num', default=None)

    parser.add_argument('--fp16', action='store_true',
                        help='Enables half precision training')

    parser.add_argument('-r', '--resume', action='store_true',
                        help='will continue training existing checkpoint')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite existing checkpoint')

    args, _ = parser.parse_known_args()

    if os.environ.get("DEBUG"):
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=level)

    _train(args)