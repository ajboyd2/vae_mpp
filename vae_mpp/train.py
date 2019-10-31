"""
Command-line utility for training a model
"""
import logging
import os
from pathlib import Path
import shutil
import sys
import argparse
import random
from collections import defaultdict

#from apex import amp
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_ as clip_grad
from tqdm import tqdm
import yaml

from vae_mpp.data import PointPatternDataset, pad_and_combine_instances
from vae_mpp.models import get_model
from vae_mpp.optim import get_optimizer, get_lr_scheduler
from vae_mpp.arguments import get_args
from vae_mpp.utils import kl_div, mmd_div, print_log


def forward_pass(args, batch, model):
    marks, timestamps, context_lengths, padding_mask \
        = batch["marks"], batch["times"], batch["context_lengths"], batch["padding_mask"]
    marks_backwards, timestamps_backwards = batch["marks_backwards"], batch["times_backwards"]

    T = 50.0  # TODO: Make this adjustable

    sample_timestamps = torch.rand_like(timestamps).clamp(min=1e-6) * T  # ~ U(0, T)

    # Forward Pass
    results = model(
        ref_marks=marks, 
        ref_timestamps=timestamps, 
        ref_marks_bwd=marks_backwards,
        ref_timestamps_bwd=timestamps_backwards,
        tgt_marks=marks, 
        tgt_timestamps=timestamps, 
        context_lengths=context_lengths,
        sample_timestamps=sample_timestamps
    )

    # Calculate losses
    ll_results = model.log_likelihood(
        return_dict=results, 
        right_window=T, 
        left_window=0.0, 
        mask=padding_mask,
    )
    log_likelihood, ll_pos_contrib, ll_neg_contrib = \
        ll_results["log_likelihood"], ll_results["positive_contribution"], ll_results["negative_contribution"]

    if args.agg_noise and args.use_encoder:
        kl_term = kl_div(results["latent_state_dict"]["mu"], 2 * results["latent_state_dict"]["log_sigma"])
        mmd_term = mmd_div(results["latent_state_dict"]["latent_state"])
    else:
        kl_term, mmd_term = torch.zeros_like(log_likelihood), torch.zeros_like(log_likelihood)

    loss = (-1 * log_likelihood) + ((1 - args.loss_alpha) * kl_term) + ((args.loss_alpha + args.loss_lambda - 1) * mmd_term)

    return {
        "loss": loss,
        "log_likelihood": log_likelihood,
        "ll_pos": ll_pos_contrib,
        "ll_neg": ll_neg_contrib,
        "kl_divergence": kl_term,
        "mmd_divergence": mmd_term,
    }

def backward_pass(args, loss, model, optimizer):
    
    optimizer.zero_grad()

    # TODO: Support different backwards passes for fp16
    loss.backward()
    
    # TODO: If using data parallel, need to perform a reduce operation
    # TODO: Update master gradients if using fp16
    clip_grad(parameters=model.parameters(), max_norm=args.grad_clip, norm_type=2)

def train_step(args, model, optimizer, lr_scheduler, batch):

    loss_results = forward_pass(args, batch, model)

    backward_pass(args, loss_results["loss"], model, optimizer)

    optimizer.step()

    lr_scheduler.step()

    return loss_results

def train_epoch(args, model, optimizer, lr_scheduler, data_loader, epoch_number):
    model.train()

    total_losses = defaultdict(lambda: 0.0)
    data_len = len(data_loader)
    for i, batch in enumerate(data_loader):
        batch_loss = train_step(args, model, optimizer, lr_scheduler, batch)
        for k, v in batch_loss.items():
            total_losses[k] += v.item()
        
        if (i+1) % args.log_interval == 0:
            print_results(args, [("LR", lr_scheduler.get_lr())] + [(k,v/args.log_interval) for k,v in total_losses.items()], epoch_number, i+1, data_len, True)
            total_losses = defaultdict(lambda: 0.0)

    if (i+1) % args.log_interval != 0:
        print_results(args, [("LR", lr_scheduler.get_lr())] + [(k,v/(i % args.log_interval)) for k,v in total_losses.items()], epoch_number, i+1, data_len, True)

def eval_step(args, model, batch):
    return forward_pass(args, batch, model)

def eval_epoch(args, model, data_loader, epoch_number):
    model.eval()
    with torch.no_grad():
        total_losses = defaultdict(lambda: 0.0)
        data_len = len(data_loader)
        for i, batch in enumerate(data_loader):
            batch_loss = eval_step(args, model, batch)
            for k, v in batch_loss.items():
                total_losses[k] += v.item()
        
    print_results(args, [(k,v/data_len) for k,v in total_losses.items()], epoch_number, i+1, data_len, False)

def print_results(args, items, epoch_number, iteration, data_len, training=True):
    msg = "[{}] Epoch {}/{} | Iter {}/{} | ".format("T" if training else "V", epoch_number, args.train_epochs, iteration, data_len)
    msg += "".join("{} {:.4E} | ".format(k, v) for k,v in items)
    print_log(msg)

def set_random_seed(args):
    """Set random seed for reproducibility."""

    seed = args.seed

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def setup_model_and_optim(args, epoch_len):
    model = get_model(
        time_embedding_size=args.time_embedding_size, 
        use_raw_time=args.use_raw_time, 
        use_delta_time=args.use_delta_time, 
        channel_embedding_size=args.channel_embedding_size,
        num_channels=args.num_channels,
        enc_hidden_size=args.enc_hidden_size,
        enc_bidirectional=args.enc_bidirectional, 
        enc_num_recurrent_layers=args.enc_num_recurrent_layers,
        agg_method=args.agg_method,
        agg_noise=args.agg_noise,
        use_encoder=args.use_encoder,
        dec_recurrent_hidden_size=args.dec_recurrent_hidden_size,
        dec_num_recurrent_layers=args.dec_num_recurrent_layers,
        dec_intensity_hidden_size=args.dec_intensity_hidden_size,
        dec_num_intensity_layers=args.dec_num_intensity_layers,
        dec_act_func=args.dec_act_func,
        dropout=args.dropout,
    )

    if args.cuda:
        model.cuda(torch.cuda.current_device())

    optimizer = get_optimizer(model, args)
    lr_scheduler = get_lr_scheduler(optimizer, args, epoch_len)

    return model, optimizer, lr_scheduler

def get_data(args):
    train_dataset = PointPatternDataset(file_path=args.train_data_path)
    valid_dataset = PointPatternDataset(file_path=args.valid_data_path)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=pad_and_combine_instances,
        drop_last=True,
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pad_and_combine_instances,
        drop_last=True,
    )

    return train_dataloader, valid_dataloader    

def save_checkpoint(args, model, optimizer, lr_scheduler):
    pass

def load_checkpoint(args, model):
    pass

def report_model_stats(model):
    encoder_parameter_count = 0
    aggregator_parameter_count = 0
    decoder_parameter_count = 0
    total = 0 
    for name, param in model.named_parameters():
        if name.startswith("encoder"):
            encoder_parameter_count += param.numel()
        elif name.startswith("aggregator"):
            aggregator_parameter_count += param.numel()
        else:
            decoder_parameter_count += param.numel()
        total += param.numel()

    print_log()
    print_log("<Parameter Counts>")
    print_log("Encoder........{}".format(encoder_parameter_count))
    print_log("Aggregator.....{}".format(aggregator_parameter_count))
    print_log("Decoder........{}".format(decoder_parameter_count))
    print_log("---Total.......{}".format(total))
    print_log()

def main():
    print_log("Getting arguments.")
    args = get_args()

    print_log("Setting seed.")
    set_random_seed(args)

    print_log("Setting up dataloaders.")
    train_dataloader, valid_dataloader = get_data(args)
    
    print_log("Setting up model, optimizer, and learning rate scheduler.")
    model, optimizer, lr_scheduler = setup_model_and_optim(args, len(train_dataloader))

    report_model_stats(model)

    print_log("Starting training.")
    for epoch in range(args.train_epochs):
        train_epoch(args, model, optimizer, lr_scheduler, train_dataloader, epoch+1)

        if (epoch % args.save_epochs == 0) or (epoch == (args.train_epochs-1)):
            save_checkpoint(args, model, optimizer, lr_scheduler)
        
        eval_epoch(args, model, valid_dataloader, epoch+1)

if __name__ == "__main__":
    main()
