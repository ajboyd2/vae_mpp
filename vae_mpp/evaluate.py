"""
Command-line utility for visualizing a model's outputs
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pickle
import torch
from torch.utils.data import DataLoader
import math
from collections import defaultdict

from vae_mpp.arguments import get_args
from vae_mpp.train import forward_pass, get_data, set_random_seed, setup_model_and_optim, load_checkpoint, report_model_stats
from vae_mpp.utils import print_log
from vae_mpp.parametric_pp import *
from vae_mpp.data import AnomalyDetectionDataset, pad_and_combine_instances

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

def l1_loss(pred_times, true_times, **kwargs):
    l1 = (pred_times.squeeze() - true_times.squeeze()).abs()
    if len(l1.shape) == 0:
        l1 = l1.unsqueeze(0)
    return l1.tolist()

def l2_loss(pred_times, true_times, **kwargs):
    l2 = (pred_times.squeeze() - true_times.squeeze()).pow(2)    
    if len(l2.shape) == 0:
        l2 = l2.unsqueeze(0)
    return l2.tolist()

# returns ranking in tensor form
def _rank(pred_dists, true_events, **kwargs):
    orig_indices = (-pred_dists).sort()[-1]  # negate so that small values are good
    all_ranks = orig_indices.sort()[-1] + 1
    ranks = all_ranks.gather(-1, true_events.unsqueeze(-1)).squeeze()
    if len(ranks.shape) == 0:
        ranks = ranks.unsqueeze(0)
    return ranks

# returns rankings in list form
def rank(pred_dists, true_events, **kwargs):
    return _rank(pred_dists, true_events).tolist()

def mean_reciprocal_rank(pred_dists, true_events, **kwargs):
    return (1 / _rank(pred_dists, true_events).float()).tolist()

def _rank_n(pred_dists, true_events, n, r=None, **kwargs):
    if r is None:
        r = _rank(pred_dists, true_events)
    return (1*(r <= n)).tolist()

def rank_1(pred_dists, true_events, **kwargs):
    return _rank_n(pred_dists, true_events, n=1, **kwargs)

def rank_5(pred_dists, true_events, **kwargs):
    return _rank_n(pred_dists, true_events, n=5, **kwargs)

def rank_10(pred_dists, true_events, **kwargs):
    return _rank_n(pred_dists, true_events, n=10, **kwargs)

def rank_25(pred_dists, true_events, **kwargs):
    return _rank_n(pred_dists, true_events, n=25, **kwargs)

def rank_100(pred_dists, true_events, **kwargs):
    return _rank_n(pred_dists, true_events, n=100, **kwargs)

all_metrics = {
    "time_l1": l1_loss,
    "time_l2": l2_loss,
    "mark_rank": rank,
    "mark_mrr": mean_reciprocal_rank,
    "mark_rank_1": rank_1,
    "mark_rank_5": rank_5,
    "mark_rank_10": rank_10,
    "mark_rank_25": rank_25,
    "mark_rank_100": rank_100,
}

def next_event_prediction(args, model, dataloader):
    model.eval()
    samples_per_time = args.samples_per_sequence
    div = 4
    num_batches = args.num_samples // (args.batch_size // div)
    num_samples = 10000
    num_samples_iterated = torch.arange(start=1, end=num_samples+1)
    base_linspace = torch.linspace(1e-10, 1.0, num_samples+1).unsqueeze(0)
    if args.cuda:
        num_samples_iterated = num_samples_iterated.cuda(torch.cuda.current_device())
        base_linspace = base_linspace.cuda(torch.cuda.current_device())

    select_condition_amounts = False  # TODO: Make this an option in the args
    if select_condition_amounts:
        events_to_cond = [2, 5, 10, 20, 50] #, 0.05, 0.0]
    else:
        # args.max_seq_len is set in the data
        events_to_cond = list(range(1, min(args.max_seq_len, 50)))

    all_results = {}
    mean_results = {}
    if select_condition_amounts:
        all_res_path = "{}/pred_task_all_results.pickle".format(args.checkpoint_path.rstrip("/"))
        mean_res_path = "{}/pred_task_mean_results.pickle".format(args.checkpoint_path.rstrip("/"))
        if os.path.exists(all_res_path) and os.path.extists(mean_res_path):
            all_results = pickle.load(open(all_results, "rb"))
            mean_results = pickle.load(open(mean_res_path, "rb"))
            events_to_cond = [i for i in events_to_cond if i not in mean_results]
    else:
        all_mean_res_path = "{}/all_pred_task_mean_results.pickle".format(args.checkpoint_path.rstrip("/"))
        if os.path.exists(all_mean_res_path):
            mean_results = pickle.load(open(all_mean_res_path, "rb"))
            events_to_cond = [i for i in events_to_cond if i not in mean_results]

    print_log(f"Next event prediction with {(args.batch_size // div) * len(dataloader)} predictions for {events_to_cond} different condition lengths.")
    print_log(f"Batch size of {args.batch_size // div} with {len(dataloader)} total batches. {num_samples} samples per prediction.")

    for cond_num in events_to_cond:
        print_log(f"Starting prediction tasks where we condition on {cond_num} events prior to prediction.")
        _, _, dataloader = get_data(args)
        #data_iter = iter(dataloader)
        results = {k:[] for k in all_metrics.keys()}
        results["pred_time"] = []
        results["true_time"] = []
        results["last_time"] = []
        #while i < num_batches:
        for i, batch in enumerate(dataloader):
            if ((i+1) % 10 == 0) or (i < 20):
                print_log(f"Batch {i+1}/{len(dataloader)} for conditioning on {cond_num} events processed.")

            #batch = next(data_iter)
            invalid_examples = batch["padding_mask"].sum(dim=-1) < (cond_num+1) 
            if ((1.0*invalid_examples).mean().item() == 1.0) or (batch["tgt_times"].shape[-1] < (cond_num+1)):
                print_log(f"Skipped batch at i={i-1}")
                continue
            else:
                batch = {k:v[~invalid_examples, ...] for k,v in batch.items()}
            #if i % (len(dataloader) // 10) == 0:

            if args.cuda:
                batch = {k:v.cuda(torch.cuda.current_device()) for k,v in batch.items()}

            ref_marks, ref_timestamps, context_lengths, padding_mask \
                = batch["ref_marks"], batch["ref_times"], batch["context_lengths"], batch["padding_mask"]
            ref_marks_backwards, ref_timestamps_backwards = batch["ref_marks_backwards"], batch["ref_times_backwards"]
            tgt_marks, tgt_timestamps = batch["tgt_marks"], batch["tgt_times"]
            pp_id = batch["pp_id"]
            T = batch["T"]  

            # truncate inputs
            true_times, true_events = tgt_timestamps[..., cond_num], tgt_marks[..., cond_num]
            
            tgt_timestamps = tgt_timestamps[..., :cond_num]
            tgt_marks = tgt_marks[..., :cond_num]
            padding_mask = padding_mask[..., :cond_num]

            last_times = tgt_timestamps[..., -1].unsqueeze(-1)  ## commented code below assumes there is no `unsqueeze(-1)` operation

            # get output intensity values
            # sample_timestamps = torch.rand(
            #     tgt_timestamps.shape[0], 
            #     num_samples, 
            #     dtype=tgt_timestamps.dtype, 
            #     device=tgt_timestamps.device
            # ).clamp(min=1e-8)  # ~ U(0,1)
            # sample_timestamps = sample_timestamps * (T.squeeze(-1) - last_times).unsqueeze(-1) + last_times.unsqueeze(-1)  # ~ U(t_{i-1}, T)
            # sample_timestamps = []
            # for i in range(last_times.shape[0]):
            #     sample_timestamps.append(torch.linspace(last_times[i]+1e-9, T[i,0], num_samples+1))
            # sample_timestamps = torch.stack(sample_timestamps, dim=0)
            sample_timestamps = base_linspace * (T - last_times) + last_times
            timestep = (T - last_times) / num_samples

            model_res = model(
                ref_marks=ref_marks, 
                ref_timestamps=ref_timestamps, 
                ref_marks_bwd=ref_marks_backwards,
                ref_timestamps_bwd=ref_timestamps_backwards,
                tgt_marks=tgt_marks, 
                tgt_timestamps=tgt_timestamps, 
                context_lengths=context_lengths,
                sample_timestamps=sample_timestamps,
                pp_id=pp_id,
            )

            sample_intensities = model_res["sample_intensities"]
            log_mark_intensity = sample_intensities["all_log_mark_intensities"]
            total_intensity = sample_intensities["total_intensity"]
            mark_prob = log_mark_intensity.exp() / total_intensity.unsqueeze(-1)
            #log_total_intensity = total_intensity.clamp(0.0001, None).log()
            #log_mark_prob = log_mark_intensity - log_total_intensity.unsqueeze(-1)
            #mark_prob = log_mark_prob.exp()

            intensity_integral = torch.cumsum(timestep * total_intensity, dim=-1)
            t_density = total_intensity * torch.exp(-intensity_integral)
            t_pit = sample_timestamps * t_density  # integrand for time estimator
            pm_pit = mark_prob * t_density.unsqueeze(-1)  # integrand for mark estimator

            # use the trapeze method of integration
            pred_times = (timestep * 0.5 * (t_pit[..., 1:] + t_pit[..., :-1])).sum(dim=-1)    # sum over sample timestep dimension
            pred_dists = (timestep.unsqueeze(-1) * 0.5 * (pm_pit[..., 1:, :] + pm_pit[..., :-1, :])).sum(dim=-2)  # sum over sample timestep dimension
            
            # MC estimate probability distributions
            # sample_intensities = model_res["sample_intensities"]
            # log_mark_intensity = sample_intensities["all_log_mark_intensities"]
            # total_intensity = sample_intensities["total_intensity"]
            # log_total_intensity = total_intensity.clamp(0.0001, None).log()
            # log_mark_prob = log_mark_intensity - log_total_intensity.unsqueeze(-1)
            # #mark_prob = log_mark_prob.exp()

            # ## p(t_i=t) = \lambda(t) exp(-\int_{t_{i-1}}^t \lambda(s) ds)
            # ## \int_{t_{i-1}}^t \lambda(s) ds \approx (t - t_{i-1}) * 1/N * \sum_{i=1}^N \lambda(s_i)
            # ##   for s_i \sim U(t_{i-1}, t]
            # cum_hazard = total_intensity.cumsum(dim=-1)
            # cum_hazard = cum_hazard * (sample_timestamps - last_times.unsqueeze(-1))
            # cum_hazard = -cum_hazard / num_samples_iterated
            # p_t = total_intensity * cum_hazard.exp() 
            # log_p_t = log_total_intensity + cum_hazard

            # ## \hat{t_i} = \int_{t_{i-1}}^T tp(t_i=t) dt
            # pred_times = (T.squeeze() - last_times) / num_samples * (sample_timestamps * p_t).sum(dim=-1)

            # ## p(k_i=k) \propto \int_{t_{i-1}}^T \lambda_k(t) / \lambda(t) * P(t_i=t) dt
            # ## since we only care about rankings, we will compute the following instead
            # ## p(k_i=k) \propto \int_{t_{i-1}}^T log \lambda_k(t) - log\lambda(t) + log P(t_i=t) dt
            # ## log_mark_prob is size (batch, num_samples, total_marks)
            # pred_dists = (log_mark_prob + log_p_t.unsqueeze(-1)).sum(dim=-2)  # sum over sample dim
            # pred_dists = pred_dists * (T.squeeze() - last_times).unsqueeze(-1) / num_samples

            # evaluate metrics
            r = _rank(pred_dists, true_events)  # compute this so we only rank them once per batch
            batch_res = {k:metric(
                pred_times=pred_times,
                pred_dists=pred_dists,
                true_times=true_times,
                true_events=true_events,
                r=r,
            ) for k,metric in all_metrics.items()}

            for t, k in zip([pred_times, true_times, last_times], ["pred_time", "true_time", "last_time"]):
                _t = t.squeeze().tolist()
                if not isinstance(_t, list):
                    _t = [_t]
                batch_res[k] = _t
            
            # import readline # optional, will allow Up/Down/History in the console
            # import code
            # variables = globals().copy()
            # variables.update(locals())
            # shell = code.InteractiveConsole(variables)
            # shell.interact()
           

            # print("DONE")
            # input()

            # store results 
            for k, b_res in batch_res.items():
                if k not in results:
                    results[k] = []
                results[k].extend(b_res)
            ## this was debugging for lastfm predictions
            ## makes no sense for other datasets
            # if any(x > 30 for x in batch_res["time_l1"]):  
            #     print_log("BAD BATCH DETECTED")
            #     print_log("BAD BATCH DETECTED")
            #     print_log("BAD BATCH DETECTED")
            #     if "bad_batches" not in results:
            #         results["bad_batches"] = []
            #     results["bad_batches"].append((batch_res, {k:v.tolist() for k,v in batch.items()}))
        
        # add to overall results

        if select_condition_amounts:
            all_results[cond_num] = results
        
        mean_res = {}    
        bad_indices = set()
        for k,v in results.items():
            if k != "bad_batches":
                bad_indices = bad_indices.union(set(i for i,el in enumerate(v) if el != el))  # filter out nan's
        for k,v in results.items():
            if k != "bad_batches":
                filtered_v = [el for i,el in enumerate(v) if i not in bad_indices]   
                if len(filtered_v) > 0:            
                    mean_res[k] = sum(filtered_v) / len(filtered_v)
                else:
                    mean_res[k] = -1
                num_seqs = len(filtered_v)
        mean_res["num_predictions"] = num_seqs
        #mean_results[cond_num] = {k:((sum(v) / len(v)) if len(v) > 0 else None) for k,v in results.items() if k != "bad_batches"}
        mean_results[cond_num] = mean_res
        # save results to file
        if select_condition_amounts:
            mean_res_path = "{}/pred_task_mean_results.pickle".format(args.checkpoint_path.rstrip("/"))
            all_res_path = "{}/pred_task_all_results.pickle".format(args.checkpoint_path.rstrip("/"))
            print_log("Saving intermittent results to", mean_res_path, all_res_path)
            pickle.dump(all_results, open(all_res_path, "wb"))
            pickle.dump(mean_results, open(mean_res_path, "wb"))
        else:
            mean_res_path = "{}/all_pred_task_mean_results.pickle".format(args.checkpoint_path.rstrip("/"))
            print_log("Saving intermittent results to", mean_res_path)
            pickle.dump(mean_results, open(mean_res_path, "wb"))

def anomaly_detection(args, model):
    model.eval()
    num_samples = 10000
    lengths_to_test = [1, 5, 10, 20, 50, None]
    labels = [f"cond_{i if i is not None else 'all'}" for i in lengths_to_test]
    all_results = {}
    for length, label in zip(lengths_to_test, labels):
        results = []

        dataset = AnomalyDetectionDataset(
            file_path=args.valid_data_path, 
            args=args, 
            max_tgt_seq_len=length,
            num_total_pairs=10000,
            test=True,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size // 8,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda x: pad_and_combine_instances(x, dataset.max_period),
            drop_last=False,
            pin_memory=args.pin_test_memory,
        )

        for i, batch in enumerate(dataloader):
            if ((i+1) % 10 == 0) or (i < 20):
                print_log(f"Batch {i+1}/{len(dataloader)} for conditioning on {length} events processed.")

            if args.cuda:
                batch = {k:v.cuda(torch.cuda.current_device()) for k,v in batch.items()}

            ref_marks, ref_timestamps, context_lengths, padding_mask \
                = batch["ref_marks"], batch["ref_times"], batch["context_lengths"], batch["padding_mask"]
            ref_marks_backwards, ref_timestamps_backwards = batch["ref_marks_backwards"], batch["ref_times_backwards"]
            tgt_marks, tgt_timestamps = batch["tgt_marks"], batch["tgt_times"]
            pp_id = batch["pp_id"]
            T = batch["T"]  

            sample_timestamps = torch.rand(
                tgt_timestamps.shape[0], 
                num_samples, 
                dtype=tgt_timestamps.dtype, 
                device=tgt_timestamps.device
            ).clamp(min=1e-8) * T # ~ U(0, T)

            model_res = model(
                ref_marks=ref_marks, 
                ref_timestamps=ref_timestamps, 
                ref_marks_bwd=ref_marks_backwards,
                ref_timestamps_bwd=ref_timestamps_backwards,
                tgt_marks=tgt_marks, 
                tgt_timestamps=tgt_timestamps, 
                context_lengths=context_lengths,
                sample_timestamps=sample_timestamps,
                pp_id=pp_id,
            )

            ll_results = model.log_likelihood(
                return_dict=model_res, 
                right_window=T, 
                left_window=0.0, 
                mask=padding_mask,
                reduce=False,
            )

            for same_source, ll in zip(batch["same_source"].tolist(), ll_results["batch_log_likelihood"].tolist()):
                results.append((same_source[0], ll))

        sorted_results = sorted(results, key=lambda x: -x[1])
        most_likely = sorted_results[:(len(sorted_results) // 2)]
        correctly_ranked = [same_source for same_source, ll in most_likely]
        proportion_ranked = sum(correctly_ranked) / len(correctly_ranked)

        all_results[label] = {
            "raw": results,
            "agg": proportion_ranked,
        }

        print_log(f"Finished anomaly detection for {length} length target sequences.")
        print_log(f"Final proportion of correctly ranked pairs: {proportion_ranked}.")
        print_log(f"Results up to now: { {k:v for k,v in all_results.items() if k == 'agg'} }")
        print_log("")

        res_path = "{}/anomaly_detection_results_{}_{}.pickle".format(args.checkpoint_path.rstrip("/"), "diff_refs" if args.anomaly_same_tgt_diff_refs else "diff_tgt", "trunc_tgt" if args.anomaly_truncate_tgts else "trunc_refs")
        print_log("Saving intermittent results to", res_path)
        pickle.dump(all_results, open(res_path, "wb"))

def _gamma_ab_to_ms(alpha, beta):
    new_mu  = {}
    new_var = {}
    for k in alpha.keys():
        a,b = alpha[k], beta[k]
        new_mu[k] = a / b
        new_var[k] = a / (b**2)

    return new_mu, new_var

def _gamma_ms_to_ab(mu, sigma_sq):
    new_alpha = {}
    new_beta = {}
    for k in mu.keys():
        m,s = mu[k], sigma_sq[k]
        new_alpha[k] = (m**2) / s
        new_beta[k]  = m / s

    return new_alpha, new_beta

def _gamma_mode(alpha, beta):
    modes = {}
    for k in alpha.keys():
        a,b = alpha[k], beta[k]
        if a < 1:
            #return None, False
            a = 1 + 1e-4
        modes[k] = (a - 1) / b
    return modes, True

def _gamma_mean(alpha, beta):
    means = {}
    for k in alpha.keys():
        a,b = alpha[k], beta[k]
        modes[k] = a / b
    return means, True

def _gamma_post(obs, alpha, beta):
    counts = defaultdict(int)
    alpha, beta = alpha.copy(), beta.copy()
    for mark in obs["ref_marks"]:
        counts[mark] += 1
    for k in alpha.keys():
        alpha[k] += counts[k]
        beta[k] += 1
    return alpha, beta

def _gamma_post_means(obs, alpha, beta, var_scale=1):
    counts = defaultdict(int)
    alpha, beta = alpha.copy(), beta.copy()
    means = {}
    for mark in obs["ref_marks"]:
        counts[mark] += 1
    for k in alpha.keys():
        means[k] = (var_scale*alpha[k] + counts[k]) / (var_scale*beta[k] + 1)
    return means

def _pois_lik(obs, lambda_mode, max_T):
    counts = defaultdict(int)
    for mark in obs["tgt_marks"]:
        counts[mark] += 1
    log_lik_i = {}
    for k, rate in lambda_mode.items():
        count = counts[k]
        adj_rate = rate * obs["T"].item() / max_T
        if rate == 0.0:
            log_lik_i[k] = -1e10
        else:
            log_lik_i[k] = -adj_rate - math.log(math.factorial(count)) + count*math.log(adj_rate)
    #return log_lik_i, sum(v for k,v in log_lik_i.items())
    return sum(v for k,v in log_lik_i.items())

def baseline_anomaly_detection(args, train_dataloader):
    train_dataset = train_dataloader.dataset

    # find mle from training data
    max_T = train_dataset.max_period
    mle_counts = defaultdict(int)
    mle_props = defaultdict(float)
    total_obs = 0
    mle_path = "{}/anomaly_detection_baseline_mle.pickle".format(args.checkpoint_path.rstrip("/"))
    if os.path.exists(mle_path):
        mle_counts, total_obs = pickle.load(open(mle_path, "rb"))
    else:
        print_log("Finding mle from training data")
        for i,obs in enumerate(train_dataset):
            if i % 50000 == 0:
                print_log(f"\tProgress {i} / {len(train_dataset)}")
            obs = {k:v.numpy() for k,v in obs.items()}
            if abs(obs["T"].item() - max_T) > 1e-2:
                continue
            
            total_obs += 1
            for mark in obs["tgt_marks"]:
                mle_counts[mark] += 1
        pickle.dump((mle_counts, total_obs), open(mle_path, "wb"))

    # tune variance on valid data
    prior_alpha = mle_counts
    prior_beta  = {k:total_obs for k in mle_counts}
    prior_mu, prior_var = _gamma_ab_to_ms(prior_alpha, prior_beta)

    var_scales = [10**(-i) for i in range(12)]
    
    valid_dataset = AnomalyDetectionDataset(
        file_path=args.valid_data_path, 
        args=args, 
        max_tgt_seq_len=None,
        num_total_pairs=1000,
        test=False,
    )
    
    acc_results = {}
    print_log(f"Testing different variance scales {var_scales}")
    for var_scale in var_scales:
        print_log(f"Trying {var_scale}")
        results = []
        #adj_prior_alpha, adj_prior_beta = _gamma_ms_to_ab(prior_mu, {k:v*var_scale for k,v in prior_var.items()})
        # adj_prior_alpha, adj_prior_beta = _gamma_adjust_priors(prior_mu, {k:v*var_scale for k,v in prior_var.items()})
        # _, good_prior = _gamma_mean(adj_prior_alpha, adj_prior_beta)
        # if not good_prior:
        #     print_log("Not a good prior")
        #     continue 

        for i,obs in enumerate(valid_dataset):
            if i % 100 == 0:
                print_log(f"\t Progress {i} / {len(valid_dataset)}")
            obs = {k:v.numpy() for k,v in obs.items()}
            # post_alpha, post_beta = _gamma_post(obs, adj_prior_alpha, adj_prior_beta, var_scale)
            # lambda_modes, _ = _gamma_mode(post_alpha, post_beta)
            #ll = _pois_lik(obs, lambda_modes, max_T)
            lambda_means = _gamma_post_means(obs, prior_alpha, prior_beta, var_scale)
            ll = _pois_lik(obs, lambda_means, max_T)
            results.append((obs["same_source"].item(), ll))

        sorted_results = sorted(results, key=lambda x: -x[1])
        #print_log(sorted_results[:100], sorted_results[-100:])
        most_likely = sorted_results[:(len(sorted_results) // 2)]
        correctly_ranked = [same_source for same_source, ll in most_likely]
        proportion_ranked = sum(correctly_ranked) / len(correctly_ranked)
        acc_results[var_scale] = proportion_ranked
        print_log(f"Var Scale {var_scale} used, resulted in {proportion_ranked} acc")

    acc_results = sorted(acc_results.items(), key=lambda x: -x[1])
    var_scale, best_valid_acc = acc_results[0]
    print_log(f"Var Scale chosen: {var_scale} w/ valid accuracy of {best_valid_acc}")
    
    lengths_to_test = [1, 5, 10, 20, 50, None][::-1]
    labels = [f"cond_{i if i is not None else 'all'}" for i in lengths_to_test]
    all_results = {}
    for length, label in zip(lengths_to_test, labels):
        print_log("Performing test on", label)

        test_dataset = AnomalyDetectionDataset(
            file_path=args.valid_data_path, 
            args=args, 
            max_tgt_seq_len=length,
            num_total_pairs=10000,
            test=True,
        )

        # get ranking 
        test_results = []
        print_log("Adjusting priors")
        adj_prior_alpha, adj_prior_beta = _gamma_ms_to_ab(prior_mu, {k:v*var_scale for k,v in prior_var.items()})
        for i,obs in enumerate(test_dataset):
            if i % 100 == 0:
                print_log(f"\t Progress {i} / {len(test_dataset)}")
            obs = {k:v.numpy() for k,v in obs.items()}
            #post_alpha, post_beta = _gamma_post(obs, adj_prior_alpha, adj_prior_beta)
            #lambda_modes,_ = _gamma_mode(post_alpha, post_beta)
            #ll = _pois_lik(obs, lambda_modes, max_T)
            lambda_means = _gamma_post_means(obs, prior_alpha, prior_beta, var_scale)
            ll = _pois_lik(obs, lambda_means, max_T)
            test_results.append((obs["same_source"].item(), ll))

        sorted_results = sorted(test_results, key=lambda x: -x[1])
        most_likely = sorted_results[:(len(sorted_results) // 2)]
        correctly_ranked = [same_source for same_source, ll in most_likely]
        proportion_ranked = sum(correctly_ranked) / len(correctly_ranked)
        
        all_results[label] = {
            "raw": test_results,
            "agg": proportion_ranked,
            "var_scale": var_scale
        }

        print_log(f"Finished anomaly detection for {length} length target sequences.")
        print_log(f"Final proportion of correctly ranked pairs: {proportion_ranked}.")
        print_log(f"Results up to now: { {k:v for k,v in all_results.items() if k == 'agg'} }")
        print_log("")

        res_path = "{}/anomaly_detection_baseline_results_{}_{}.pickle".format(args.checkpoint_path.rstrip("/"), "diff_refs" if args.anomaly_same_tgt_diff_refs else "diff_tgt", "trunc_tgt" if args.anomaly_truncate_tgts else "trunc_refs")
        print_log("Saving intermittent results to", res_path)
        pickle.dump(all_results, open(res_path, "wb"))
    

def main():
    print_log("Getting arguments.")
    args = get_args()

    # args.anomaly_detection = True
    # args.anomaly_same_tgt_diff_refs = True  # default is True, True, False
    # args.anomaly_truncate_tgts = False
    # args.anomaly_truncate_refs = True 
    args.sample_generations = True
    args.top_k = 0
    args.top_p = 0

    if args.visualize or args.sample_generations:
        args.batch_size = 4
    if args.get_latents:
        args.shuffle = False
        args.same_tgt_and_ref = True
    else:
        args.shuffle = False

    if not (args.next_event_prediction or args.anomaly_detection):
        args.train_data_path = [fp.replace("train", "vis" if args.visualize else "valid") for fp in args.train_data_path]

    print_log("Setting seed.")
    set_random_seed(args)

    print_log("Setting up dataloaders.")
    args.pin_test_memory = True
    # train_dataloader contains the right data for most tasks
    train_dataloader, valid_dataloader, test_dataloader = get_data(args)
    
    print_log("Setting up model, optimizer, and learning rate scheduler.")
    model, _, _ = setup_model_and_optim(args, len(train_dataloader))

    report_model_stats(model)

    load_result = load_checkpoint(args, model)
    if load_result == 0:
        old_path = args.checkpoint_path
        args.checkpoint_path = old_path.rstrip("/") + "/data_ablation/"
        print_log(f"Model not found in {old_path}.")
        print_log(f"Trying to load model instead from {args.checkpoint_path}.")
        load_checkpoint(args, model)
        args.checkpoint_path = old_path

    if args.visualize:
        print_log("Starting visualization.")
        save_and_vis_intensities(args, model, train_dataloader)
    elif args.sample_generations:
        print_log("Sampling generations.")
        sample_generations(args, model, test_dataloader)  # train_dataloader)
    elif args.likelihood_over_time:
        print_log("Starting likelihood over time analysis.")
        if "amazon" in args.checkpoint_path:
            args.likelihood_resolution = args.likelihood_resolution / 4.0  # 1/4 day resolution
        elif "lastfm" in args.checkpoint_path:
            args.likelihood_resolution = args.likelihood_resolution / 6.0  # 10 minute resolution
        # else: 1 hour resolution over 1 week = 168 bins

        likelihood_over_time(args, model, test_dataloader)  # train_dataloader)
    elif args.get_latents:
        print_log("Extracting latent states.")
        save_latents(args, model, train_dataloader)
    elif args.anomaly_detection:
        print_log("Starting anomaly detection experiments.")
        with torch.no_grad():
            anomaly_detection(args, model)
        if "rmtpp" in args.checkpoint_path:
            baseline_anomaly_detection(args, train_dataloader)
    elif args.next_event_prediction:
        print_log("Performing next event prediction experiments.")
        #args.num_workers = 0
        args.num_samples = (len(test_dataloader)-1) * args.batch_size
        with torch.no_grad():
            next_event_prediction(args, model, test_dataloader)

if __name__ == "__main__":
    main()
