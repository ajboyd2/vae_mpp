"""
Command-line utility for visualizing a model's outputs
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pickle
import torch
import math

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

        actual_times = batch["tgt_times"].squeeze().tolist()
        actual_marks = batch["tgt_marks"].squeeze().tolist()
        
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
    partial_sum = torch.where(num_valid != 0, partial_sum, torch.zeros_like(partial_sum))
    num_valid = torch.where(num_valid != 0, num_valid, torch.zeros_like(num_valid) + 1)
    return partial_sum, partial_sum / num_valid, (partial_sum / num_valid * T)

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
    ce_total_contributions = {}
    overall_freq = {}
    lik_diff_contributions = {}
    pos_diff_contributions = {}
    neg_diff_contributions = {}
    ce_diff_contributions = {}

    all_contributions = {
        "lik_total": lik_total_contributions, 
        "pos_total": pos_total_contributions, 
        "neg_total": neg_total_contributions, 
        "ce_total": ce_total_contributions,
    }

    res = args.likelihood_resolution 
    model.eval()

    for i, batch in enumerate(dataloader):
        if i % 20 == 0:
            print_log("Progress: {} / {}".format(i, len(dataloader)))
        with torch.no_grad():
            ll_results, sample_timestamps, tgt_timestamps = forward_pass(args, batch, model, sample_timestamps=None, num_samples=1000, get_raw_likelihoods=True)
            pos_cont, neg_cont, ce = ll_results["positive_contribution"], ll_results["negative_contribution"], ll_results["cross_entropy"]
            
            prev_lik, prev_pos, prev_neg, prev_count, prev_ce = 0, 0, 0, 0, 0
            for T in np.arange(res, batch["T"].max().item() + res, res):
                partial_pos_sum, partial_pos_mean, partial_pos_mean_scaled = partial_neg_contributions(pos_cont, tgt_timestamps, T)
                partial_ce_sum,  partial_ce_mean,  partial_ce_mean_scaled  = partial_neg_contributions(ce, tgt_timestamps, T)
                partial_neg_sum, partial_neg_mean, partial_neg_mean_scaled = partial_neg_contributions(neg_cont, sample_timestamps, T)
                partial_lik_cont = partial_pos_sum - partial_neg_mean_scaled
                new_conts = {
                    "lik_total": partial_lik_cont,
                    "pos_total": partial_ce_mean + partial_pos_mean,
                    "neg_total": partial_neg_mean,
                    "ce_total":  partial_ce_mean,
                }
                for key, new_cont in new_conts.items():
                    add_contribution(all_contributions[key], new_cont, T, batch["T"])


    mean_contributions = {}
    lower_ci_contributions = {}
    upper_ci_contributions = {}
    for key, total_contributions in all_contributions.items():
        if "ce_" in key:
            mean_contributions[key] = sorted([(t,sum(ls) / sum(1 for x in ls if (x != 0) and (x != 0.0))) for t,ls in total_contributions.items()])
        elif "pos_" in key:
            mean_contributions[key] = sorted([(t,sum(ls) / sum(1 for x in all_contributions["ce_total"][t] if (x != 0) and (x != 0.0))) for t,ls in total_contributions.items()])
        else:
            mean_contributions[key] = sorted([(t,sum(ls) / len(ls)) for t,ls in total_contributions.items()])

    pickle.dump(
        {"mean": mean_contributions}, 
        open("{}/likelihood_data.pickle".format(args.checkpoint_path.rstrip("/")), "wb"),
    )

def sample_generations(args, model, dataloader):
    model.eval()
    samples_per_time = args.samples_per_sequence
    users_sampled = args.num_samples
    T_pcts = [0.5, 0.3, 0.1] #, 0.05, 0.0]

    all_samples = []
    data_iter = iter(dataloader)

    i = 0
    while i < users_sampled:
#    for i, batch in enumerate(dataloader):
#        if i >= users_sampled:
#            break
        try:
            batch = next(data_iter)
            print_log("New user {}".format(i))

            if args.cuda:
                batch = {k:v.cuda(torch.cuda.current_device()) for k,v in batch.items()}

            ref_marks, ref_timestamps, context_lengths, padding_mask \
                = batch["ref_marks"], batch["ref_times"], batch["context_lengths"], batch["padding_mask"]
            ref_marks_backwards, ref_timestamps_backwards = batch["ref_marks_backwards"], batch["ref_times_backwards"]
            tgt_marks, tgt_timestamps = batch["tgt_marks"], batch["tgt_times"]
            pp_id = batch["pp_id"]
            tgt_timestamps = tgt_timestamps[..., :padding_mask.cumsum(-1).max().item()]
            tgt_marks = tgt_marks[..., :padding_mask.cumsum(-1).max().item()]

            T = batch["T"]  

            user_samples = {"original_times": tgt_timestamps.squeeze().tolist(), "original_marks": tgt_marks.squeeze().tolist(), "original_T": T.squeeze().tolist(), "samples": {}}

            for pct in T_pcts:
                print_log("New pct {}".format(pct))
                user_samples["samples"][pct] = []
                if pct == 0.0:
                    new_tgt_timestamps = tgt_timestamps[..., :1]*10000
                    new_tgt_marks = tgt_marks[..., :1]
                    left_window = 0.0
                else:
                    new_tgt_timestamps = tgt_timestamps[..., :math.floor(pct * tgt_timestamps.shape[-1])+1] #torch.where(good_times, tgt_timestamps, torch.ones_like(tgt_timestamps) * 10000)
                    new_tgt_marks = tgt_marks[..., :math.floor(pct * tgt_timestamps.shape[-1])+1]
                    left_window = new_tgt_timestamps[..., -1].squeeze().item()

                for j in range(samples_per_time):
                    print("New sample {}".format(j))
                    samples = None
                    m = 1.0
                    while samples is None:
                        if m >= 10.0:
                            break
                        samples = model.sample_points(
                            ref_marks=ref_marks, 
                            ref_timestamps=ref_timestamps, 
                            ref_marks_bwd=ref_marks_backwards, 
                            ref_timestamps_bwd=ref_timestamps_backwards, 
                            tgt_marks=new_tgt_marks, 
                            tgt_timestamps=new_tgt_timestamps, 
                            context_lengths=context_lengths, 
                            dominating_rate=args.dominating_rate * m, 
                            T=T,
                            left_window=left_window,
                            top_k=args.top_k,
                            top_p=args.top_p,
                        )
                        m *= 1.5

                    if samples is None:
                        print("No good sample found. Skipping")
                        continue

                    sampled_times, sampled_marks = samples

                    held_out_marks = set(tgt_marks[...,math.floor(pct * tgt_timestamps.shape[-1]):].squeeze().tolist())

                    print("Pct: {} | Left Window: {} |Num Original: {} | Num Conditioned: {} | Num Sampled Alone: {} | Unique Marks on Held Out: {} | Unique Marks Sampled: {} | Common Marks: {}".format(
                        pct,
                        left_window,
                        tgt_timestamps.squeeze().shape[0],
                        math.floor(pct * tgt_timestamps.shape[-1]),
                        len(sampled_times),
                        len(held_out_marks),
                        len(set(sampled_marks)),
                        len(held_out_marks.intersection(set(sampled_marks))),
                    ))
                    assert(len(sampled_times) == 0 or left_window <= min(sampled_times))
                    user_samples["samples"][pct].append((sampled_times, sampled_marks))
            
            all_samples.append(user_samples)
            i += 1
        except StopIteration:
            break  # ran out of data
        except:
            continue  # data processing error


    pickle.dump(all_samples, open("{}/scaling_samples_top_p_{}_top_k_{}.pickle".format(args.checkpoint_path.rstrip("/"), args.top_p, args.top_k), "wb"))

def save_latents(args, model, dataloader):
    num_samples = len(dataloader) 
    model.eval() 
    latents = []
    for i,batch in enumerate(dataloader):
        if args.cuda:
            batch = {k:v.cuda(torch.cuda.current_device()) for k,v in batch.items()}
        if i % (num_samples // 10) == 0:
            print_log("{} Latent state batches extracted".format(i))
        if i > num_samples:
            break
        ref_marks, ref_timestamps, context_lengths, padding_mask \
            = batch["ref_marks"], batch["ref_times"], batch["context_lengths"], batch["padding_mask"]
        ref_marks_backwards, ref_timestamps_backwards = batch["ref_marks_backwards"], batch["ref_times_backwards"]
        pp_id = batch["pp_id"]      
        with torch.no_grad():
            latent = model.get_latent(
                ref_marks_fwd=ref_marks,
                ref_timestamps_fwd=ref_timestamps,
                ref_marks_bwd=ref_marks_backwards,
                ref_timestamps_bwd=ref_timestamps_backwards,
                context_lengths=context_lengths,
                pp_id=pp_id,
            )
        mean = latent["latent_state"]
        sigma = latent["q_z_x"]
        if sigma is None:
            sigma = torch.zeros_like(mean)
        else:
            sigma = sigma.scale

        for ls, sm, cl, m, t, pp in zip(mean.tolist(), sigma.tolist(), context_lengths.squeeze().tolist(), ref_marks.tolist(), ref_timestamps.tolist(), pp_id.squeeze().tolist()):
            m, t = m[:cl+1], t[:cl+1]
            mark_counts = {}
            t_delta = [t1 - t0 for t1, t0 in zip(t[1:], t[:-1])]
            if len(t_delta) == 0:
                continue
            for k in m:
                if k not in mark_counts:
                    mark_counts[k] = 1
                else:
                    mark_counts[k] += 1
            latents.append({
                "latent_mu": ls,
                "latent_sigma": sm,
                "mark_counts": mark_counts,
                "total_events": len(m),
                "mean_inter_event_time": sum(t_delta) / len(t_delta),
                "median_inter_event_time": sorted(t_delta)[len(t_delta) // 2],
                "user_id": pp,
            })

    pickle.dump(latents, open("{}/extracted_latents.pickle".format(args.checkpoint_path.rstrip("/")), "wb"))


def main():
    print_log("Getting arguments.")
    args = get_args()

    if args.visualize or args.sample_generations:
        args.batch_size = 1
    if args.get_latents:
        args.shuffle = False
        args.same_tgt_and_ref = True
    else:
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
    elif args.sample_generations:
        print_log("Sampling generations.")
        sample_generations(args, model, dataloader)
    elif args.likelihood_over_time:
        print_log("Starting likelihood over time analysis.")
        likelihood_over_time(args, model, dataloader)
    elif args.get_latents:
        print_log("Extracting latent states.")
        save_latents(args, model, dataloader)

if __name__ == "__main__":
    main()
