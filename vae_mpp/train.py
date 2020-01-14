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
from sklearn.linear_model import LogisticRegression
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


def forward_pass(args, batch, model, sample_timestamps=None):
    if args.cuda:
        batch = {k:v.cuda(torch.cuda.current_device()) for k,v in batch.items()}

    marks, timestamps, context_lengths, padding_mask \
        = batch["marks"], batch["times"], batch["context_lengths"], batch["padding_mask"]
    marks_backwards, timestamps_backwards = batch["marks_backwards"], batch["times_backwards"]
    pp_id = batch["pp_id"]

    T = batch["T"]  

    if sample_timestamps is None:
        sample_timestamps = torch.rand(timestamps.shape[0], 150, dtype=timestamps.dtype, device=timestamps.device).clamp(min=1e-8) * T #torch.rand_like(timestamps).clamp(min=1e-8) * T # ~ U(0, T)

    # Forward Pass
    results = model(
        ref_marks=marks, 
        ref_timestamps=timestamps, 
        ref_marks_bwd=marks_backwards,
        ref_timestamps_bwd=timestamps_backwards,
        tgt_marks=marks, 
        tgt_timestamps=timestamps, 
        context_lengths=context_lengths,
        sample_timestamps=sample_timestamps,
        pp_id=pp_id,
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
        #kl_term = kl_div(results["latent_state_dict"]["mu"], results["latent_state_dict"]["log_var"])
        kl_term = kl_div(results["q_z_x"], results["p_z"]).mean()
        #mmd_term = mmd_div(results["latent_state_dict"]["latent_state"])
    else:
        #kl_term, mmd_term = torch.zeros_like(log_likelihood), torch.zeros_like(log_likelihood)
        kl_term = torch.zeros_like(log_likelihood)

    objective = log_likelihood - (args.loss_beta * kl_term) #- (args.loss_lambda * mmd_term)
    loss = -1 * objective  # minimize loss, maximize objective

    #print_log("Beta: {:.4f} | Lambda: {:.4f}".format(args.loss_beta, args.loss_lambda))

    return {
        "loss": loss,
        "log_likelihood": log_likelihood,
        "ll_pos": ll_pos_contrib,
        "ll_neg": ll_neg_contrib,
        "kl_divergence": kl_term,
        #"mmd_divergence": mmd_term,
    }, results

def backward_pass(args, loss, model, optimizer):
    
    optimizer.zero_grad()

    # TODO: Support different backwards passes for fp16
    loss.backward()
    
    # TODO: If using data parallel, need to perform a reduce operation
    # TODO: Update master gradients if using fp16
    clip_grad(parameters=model.parameters(), max_norm=args.grad_clip, norm_type=2)

def train_step(args, model, optimizer, lr_scheduler, batch):

    loss_results, _ = forward_pass(args, batch, model)

    backward_pass(args, loss_results["loss"], model, optimizer)

    optimizer.step()

    lr_scheduler.step()

    return loss_results

def train_epoch(args, model, optimizer, lr_scheduler, dataloader, epoch_number):
    model.train()

    total_losses = defaultdict(lambda: 0.0)
    data_len = len(dataloader)
    for i, batch in enumerate(dataloader):
        batch_loss = train_step(args, model, optimizer, lr_scheduler, batch)
        for k, v in batch_loss.items():
            total_losses[k] += v.item()
        
        if (i+1) % args.log_interval == 0:
            items_to_print = [("LR", lr_scheduler.get_lr())]
            items_to_print.extend([(k,v/args.log_interval) for k,v in total_losses.items()])
            items_to_print.extend([("beta", args.loss_beta), ("lambda", args.loss_lambda)])
            print_results(args, items_to_print, epoch_number, i+1, data_len, True)
            total_losses = defaultdict(lambda: 0.0)

    if (i+1) % args.log_interval != 0:
        items_to_print = [("LR", lr_scheduler.get_lr())]
        items_to_print.extend([(k,v/(i % args.log_interval)) for k,v in total_losses.items()])
        items_to_print.extend([("beta", args.loss_beta), ("lambda", args.loss_lambda)])
        print_results(args, items_to_print, epoch_number, i+1, data_len, True)

def eval_step(args, model, batch):
    return forward_pass(args, batch, model)

def eval_epoch(args, model, valid_dataloader, train_dataloader, epoch_number):
    model.eval()

    with torch.no_grad():
        total_losses = defaultdict(lambda: 0.0)
        data_len = len(valid_dataloader)
        valid_latents, valid_labels = [], []
        for i, batch in enumerate(valid_dataloader):
            batch_loss, results = eval_step(args, model, batch)
            if args.classify_latents:
                valid_latents.append(results["latent_state_dict"]["latent_state"])
                valid_labels.append(batch["pp_id"])
            for k, v in batch_loss.items():
                total_losses[k] += v.item()

    print_results(args, [(k,v/data_len) for k,v in total_losses.items()], epoch_number, i+1, data_len, False)

    if args.classify_latents:
        with torch.no_grad():
            train_latents, train_labels = [], []
            for batch in train_dataloader:
                _, results = eval_step(args, model, batch)
                train_latents.append(results["latent_state_dict"]["latent_state"])
                train_labels.append(batch["pp_id"])

        train_latents = torch.cat(train_latents, dim=0).squeeze().numpy()
        train_labels = torch.cat(train_labels, dim=0).squeeze().numpy()
        valid_latents = torch.cat(valid_latents, dim=0).squeeze().numpy()
        valid_labels = torch.cat(valid_labels, dim=0).squeeze().numpy()
        clf = LogisticRegression(
            random_state=args.seed, 
            solver="liblinear",
            multi_class="auto",
        ).fit(train_latents, train_labels)

        train_acc, valid_acc = clf.score(train_latents, train_labels), clf.score(valid_latents, valid_labels)

        t_vals, t_counts = np.unique(train_labels, return_counts=True)
        t_most_freq_val, t_most_freq_count = t_vals[t_counts.argmax()], t_counts.max()
        naive_train_acc = t_most_freq_count / len(train_labels)

        v_vals, v_counts = np.unique(valid_labels, return_counts=True)
        v_most_freq_count = v_counts[np.where(v_vals == t_most_freq_val)[0][0]]
        naive_valid_acc = v_most_freq_count / len(valid_labels) 

        print_log("[C] Epoch {}/{} | Train Acc {:.4E} | Valid Acc {:.4E} | (N) Train Acc {:.4E} | (N) Valid Acc {:.4E}".format(
            epoch_number, args.train_epochs,
            train_acc, valid_acc,
            naive_train_acc, naive_valid_acc,
        ))
        
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
        max_period=args.max_period,
        channel_embedding_size=args.channel_embedding_size,
        num_channels=args.num_channels,
        enc_hidden_size=args.enc_hidden_size,
        enc_bidirectional=args.enc_bidirectional, 
        enc_num_recurrent_layers=args.enc_num_recurrent_layers,
        latent_size=args.latent_size,
        agg_method=args.agg_method,
        agg_noise=args.agg_noise,
        use_encoder=args.use_encoder,
        dec_recurrent_hidden_size=args.dec_recurrent_hidden_size,
        dec_num_recurrent_layers=args.dec_num_recurrent_layers,
        dec_intensity_hidden_size=args.dec_intensity_hidden_size,
        dec_intensity_factored_heads=args.dec_intensity_factored_heads,
        dec_num_intensity_layers=args.dec_num_intensity_layers,
        dec_intensity_use_embeddings=args.dec_intensity_use_embeddings,
        dec_act_func=args.dec_act_func,
        dropout=args.dropout,
        amortized=args.amortized,
    )

    if args.cuda:
        model.cuda(torch.cuda.current_device())

    optimizer = get_optimizer(model, args)
    lr_scheduler = get_lr_scheduler(optimizer, args, epoch_len)

    return model, optimizer, lr_scheduler

def get_data(args):
    train_dataset = PointPatternDataset(file_path=args.train_data_path)
    args.num_channels = train_dataset.vocab_size

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        collate_fn=pad_and_combine_instances,
        drop_last=True,
    )

    args.max_period = train_dataset.get_max_T() / 2.0

    print_log("Loaded {} / {} training examples / batches from {}".format(len(train_dataset), len(train_dataloader), args.train_data_path))

    if args.do_valid:
        valid_dataset = PointPatternDataset(file_path=args.valid_data_path)

        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            collate_fn=pad_and_combine_instances,
            drop_last=True,
        )
        print_log("Loaded {} / {} validation examples / batches from {}".format(len(valid_dataset), len(valid_dataloader), args.valid_data_path))
    else:
        valid_dataloader = None


    return train_dataloader, valid_dataloader    

def save_checkpoint(args, model, optimizer, lr_scheduler, epoch):
    # Create folder if not already created
    folder_path = args.checkpoint_path
    folders = folder_path.split("/")
    for i in range(len(folders)):
        if folders[i] == "":
            continue
        intermediate_path = "/".join(folders[:i+1])
        if not os.path.exists(intermediate_path):
            os.mkdir(intermediate_path)

    final_path = "{}/model_{}.pt".format(folder_path.rstrip("/"), epoch)
    if os.path.exists(final_path):
        os.remove(final_path)
    torch.save(model.state_dict(), final_path)
    print_log("Saved model at {}".format(final_path))

def load_checkpoint(args, model):
    folder_path = args.checkpoint_path
    files = [f for f in os.listdir(folder_path) if ".pt" in f]
    file_path = "{}/{}".format(folder_path.rstrip("/"), sorted(files)[-1])

    model.load_state_dict(torch.load(file_path))
    print_log("Loaded model from {}".format(file_path))

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

    if args.finetune:
        load_checkpoint(args, model)

    print_log("Starting training.")

    if args.do_valid:
        eval_epoch(args, model, valid_dataloader, train_dataloader, 0)

    for epoch in range(args.train_epochs):
        train_epoch(args, model, optimizer, lr_scheduler, train_dataloader, epoch+1)

        if (epoch % args.save_epochs == 0) or (epoch == (args.train_epochs-1)):
            save_checkpoint(args, model, optimizer, lr_scheduler, epoch)
        
        if args.do_valid:
            eval_epoch(args, model, valid_dataloader, train_dataloader, epoch+1)

if __name__ == "__main__":
    main()
