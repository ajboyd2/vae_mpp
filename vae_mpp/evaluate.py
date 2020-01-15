"""
Command-line utility for visualizing a model's outputs
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pickle
import torch

from vae_mpp.arguments import get_args
from vae_mpp.train import forward_pass, get_data, set_random_seed, setup_model_and_optim, load_checkpoint, report_model_stats
from vae_mpp.utils import print_log
from vae_mpp.parametric_pp import *


def save_and_vis_intensities(args, model, dataloader):
    model.eval()
    colors = sns.color_palette()
    pp_objs = None
    for i, batch in enumerate(dataloader):
        if args.cuda:
            batch = {k:v.cuda(torch.cuda.current_device()) for k,v in batch.items()}
        T = batch["T"][0].item()
        '''
        if "pp_id" in batch:
            pp_obj_idx = batch["pp_id"][0].item()
            if pp_objs is None:
                _pp_objs = pickle.load(open("/mnt/c/Users/Alex/Research/vae_mpp/data/15_pps/interesting_pp_objs_dicts.pickle", "rb"))
                pp_objs = []
                for _pp in _pp_objs:
                    pp = SelfExcitingProcess(3)
                    pp.mu = _pp["mu"]
                    pp.alpha = _pp["alpha"]
                    pp.delta = _pp["delta"]
                    pp_objs.append(pp)
        '''
        all_times = torch.linspace(0, T, 500, dtype=torch.float, device=batch["T"].device).unsqueeze(0) + 1e-8

        def to_cpu(obj):
            if isinstance(obj, dict):
                return {k:to_cpu(v) for k,v in obj.items()}
            elif isinstance(obj, list):
                return [to_cpu(v) for v in obj]
            elif isinstance(obj, torch.distributions.distribution.Distribution):
                return obj
            else:
                return obj.cpu()

        # Process Batch
        with torch.no_grad():
            losses, results = forward_pass(args, batch, model, sample_timestamps=all_times)
            results = to_cpu(results)
            all_times = to_cpu(all_times)

        print(i, results["latent_state"].tolist())

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12,8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        fig.tight_layout()
        all_times = all_times.squeeze().numpy()
       
        model_intensities = torch.exp(results["sample_intensities"]["all_log_mark_intensities"]).squeeze()
        total_intensity = results["sample_intensities"]["total_intensity"].squeeze().numpy()
        model_intensities = model_intensities.numpy()
        print(model_intensities.shape)

        for k in range(model_intensities.shape[1]):
            ax_top.plot(all_times, model_intensities[:, k], color=colors[k], label="Model - k={}".format(k))

        #ax_top.plot(all_times, total_intensity, linestyle="dashed", color="black", label="Model - Total")

        actual_times = batch["times"].squeeze().tolist()
        actual_marks = batch["marks"].squeeze().tolist()
        
        if pp_objs is not None:
            pp_obj = pp_objs[pp_obj_idx]
            pp_obj.clear()
            for t,m in zip(actual_times, actual_marks):
                pp_obj.update(t, m, 0)
            actual_intensities = np.array([pp_obj.intensity(t=t, batch=0).squeeze() for t in all_times])

            for k in range(model_intensities.shape[1]):
                ax_top.plot(all_times, actual_intensities[:, k], color=colors[k], alpha=0.5, linestyle='dashed', label="Real  - k={}".format(k))

        events = []
        for _ in range(model_intensities.shape[1]):
            events.append([])

        for pt, mark in zip(actual_times, actual_marks):
            events[mark].append(pt)
            ax_top.axvline(pt, color=colors[mark], alpha=0.2, linestyle='dotted', linewidth=1)
        ax_bot.eventplot(events, colors=colors[:model_intensities.shape[1]], linelengths=0.4, linewidths=1.2)

        ax_top.set_ylabel("Intensity by Mark")
        ax_bot.set_ylabel("Events")
        ax_bot.get_yaxis().set_ticks([])
        ax_bot.set_xlabel("Time")
        ax_top.set_xlim((0, T))
        ax_bot.set_xlim((0, T))
        ax_top.legend()
        final_path = "{}/example_{}.png".format(args.checkpoint_path.rstrip("/"), i)
        if os.path.exists(final_path):
            os.remove(final_path)
        fig.savefig(final_path,
                    dpi=150,
                    bbox_inches="tight")

def filter_contributions(cont, times, T):
    valid_times = times <= T
    num_valid = valid_times.sum(-1)
    partial_sum = torch.where(valid_times, cont, torch.zeros_like(cont)).sum(-1)
    return partial_sum, num_valid

def partial_pos_contributions_and_count(cont, times, T):
    partial_sum, num_valid = filter_contributions(cont, times, T)
    return partial_sum, num_valid

def partial_neg_contributions(cont, times, T):
    partial_sum, num_valid = filter_contributions(cont, times, T)
    partial_sum = torch.where(num_valid == 0, partial_sum, torch.zeros_like(partial_sum))
    num_valid = torch.where(num_valid == 0, num_valid, torch.zeros_like(num_valid) + 1)
    return (partial_sum / num_valid * T)

def add_contribution(total_contributions, new_conts, T, T_limits):
    for new_cont_tensor, T_limit_tensor in zip(new_conts, T_limits):
        new_cont, T_limit = new_cont_tensor.item(), T_limit_tensor.item()
        if T <=T_limit:
            if T in total_contributions:
                total_contributions[T].append(new_cont)
            else:
                total_contributions[T] = [new_cont]

def likelihood_over_time(args, model, dataloader):

    lik_total_contributions = {}
    pos_total_contributions = {}
    neg_total_contributions = {}
    overall_freq = {}
    lik_diff_contributions = {}
    pos_diff_contributions = {}
    neg_diff_contributions = {}

    all_contributions = {
        "lik_total": lik_total_contributions, 
        "pos_total": pos_total_contributions, 
        "neg_total": neg_total_contributions, 
        "lik_diff": lik_diff_contributions, 
        "pos_diff": pos_diff_contributions, 
        "neg_diff": neg_diff_contributions, 
        "overall_freq": overall_freq,
    }

    res = args.likelihood_resolution
    model.eval()

    for i, batch in enumerate(dataloader):
        if i % 20 == 0:
            print_log("Progress: {} / {}".format(i, len(dataloader)))
        with torch.no_grad():
            ll_results, sample_timestamps, tgt_timestamps = forward_pass(args, batch, model, sample_timestamps=None, num_samples=1000, get_raw_likelihoods=True)
            pos_cont, neg_cont = ll_results["positive_contribution"], ll_results["negative_contribution"]
            #sample_timestamps, tgt_timestamps, pos_cont, neg_cont = sample_timestamps.squeeze(), tgt_timestamps.squeeze(), pos_cont.squeeze(), neg_cont.squeeze()
            
            prev_lik, prev_pos, prev_neg, prev_count = 0, 0, 0, 0
            for T in np.arange(0, batch["T"].max().item() + res, res):
                partial_pos_cont, new_count = partial_pos_contributions_and_count(pos_cont, tgt_timestamps, T)
                partial_neg_cont = partial_neg_contributions(neg_cont, sample_timestamps, T)
                partial_lik_cont = partial_pos_cont - partial_neg_cont
                new_conts = {
                    "lik_total": partial_lik_cont,
                    "pos_total": partial_pos_cont,
                    "neg_total": partial_neg_cont,
                    "lik_diff": partial_lik_cont - prev_lik,
                    "pos_diff": partial_pos_cont - prev_pos,
                    "neg_diff": partial_neg_cont - prev_neg,
                    "overall_freq": new_count - prev_count,
                }
                for key, new_cont in new_conts.items():
                    add_contribution(all_contributions[key], new_cont, T, batch["T"])

                prev_lik, prev_pos, prev_neg, prev_count = partial_lik_cont, partial_pos_cont, partial_neg_cont, new_count

    mean_contributions = {}
    for key, total_contributions in all_contributions.items():
        mean_contributions[key] = sorted([(t,sum(ls) / len(ls)) for t,ls in total_contributions.items()])

    pickle.dump(
        {"raw": all_contributions, "mean": mean_contributions},
        open("{}/likelihood_data.pickle".format(args.checkpoint_path.rstrip("/")), "wb"),
    )

def main():
    print_log("Getting arguments.")
    args = get_args()

    if args.visualize:
        args.batch_size = 1
    args.shuffle = False
    args.train_data_path = [fp.replace("train", "vis" if args.visualize else "valid") for fp in args.train_data_path]

    print_log("Setting seed.")
    set_random_seed(args)

    print_log("Setting up dataloaders.")
    dataloader, _ = get_data(args)
    
    print_log("Setting up model, optimizer, and learning rate scheduler.")
    model, _, _ = setup_model_and_optim(args, len(dataloader))

    report_model_stats(model)

    load_checkpoint(args, model)

    if args.visualize:
        print_log("Starting visualization.")
        save_and_vis_intensities(args, model, dataloader)
    elif args.likelihood_over_time:
        print_log("Starting likelihood over time analysis.")
        likelihood_over_time(args, model, dataloader)

if __name__ == "__main__":
    main()
