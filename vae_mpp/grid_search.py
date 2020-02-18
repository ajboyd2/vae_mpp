import pickle
import itertools
import math
import time
import os

from vae_mpp.arguments import get_args
from vae_mpp.train import main as single_run

if __name__ == "__main__":
    args = get_args()
    args.save_epochs = 1000
    base_path = args.checkpoint_path

    #base_path = "/mnt/c/Users/Alex/Research/scale/nlp_takehome"
    #eval_seq_lengths = [42, 50, 60, 70, 80, 90, 100]
    #args.eval_seq_lengths = eval_seq_lengths

    #hidden_sizes = [1<<3, 1<<4, 1<<5, 1<<6, 1<<7]
    #num_layers = [1, 2]
    #bidirectional = [True, False]
    #train_samples = [1<<14, 1<<16, 1<<18]
    #args.train_epochs=1
    orig_epochs = args.train_epochs
    #'''
    opts = [
#        {"train_data_percentage": 0.01}, 
#        {"train_data_percentage": 0.05}, 
        {"train_data_percentage": 0.1, "save_epochs": 6, "valid_epochs": 6,}, 
        {"train_data_percentage": 0.2, "save_epochs": 6, "valid_epochs": 6,}, 
        {"train_data_percentage": 0.3, "save_epochs": 4, "valid_epochs": 4,}, 
        {"train_data_percentage": 0.5, "save_epochs": 4, "valid_epochs": 4,}, 
        {"train_data_percentage": 0.7, "save_epochs": 2, "valid_epochs": 2,}, 
        {"train_data_percentage": 0.9, "save_epochs": 2, "valid_epochs": 2,}, 
#        {"train_data_percentage": 0.95}, 
#        {"train_data_percentage": 0.99}, 
        {"train_data_percentage": 1.0, "save_epochs": 2, "valid_epochs": 2,},
    ]; name = "pct"
    orig_batch_size = args.batch_size
    args.finetune = True
    args.save_epochs = 2
    #args.seed = 12341234
    args.valid_epochs = 2
    args.early_stop = True
    #'''
    '''
    args.train_epochs = 15
    opts = [
        {"time_embedding_size": 1, "use_raw_time": True, "use_delta_time": False,},
        {"time_embedding_size": 1, "use_raw_time": False, "use_delta_time": True,},
        {"time_embedding_size": 8, "use_raw_time": True, "use_delta_time": False,},
        {"time_embedding_size": 8, "use_raw_time": False, "use_delta_time": True,},
        {"time_embedding_size": 16, "use_raw_time": True, "use_delta_time": False,},
        {"time_embedding_size": 16, "use_raw_time": False, "use_delta_time": True,},
        {"time_embedding_size": 32, "use_raw_time": True, "use_delta_time": False,},
        {"time_embedding_size": 32, "use_raw_time": False, "use_delta_time": True,},
        {"time_embedding_size": 64, "use_raw_time": True, "use_delta_time": False,},
        {"time_embedding_size": 64, "use_raw_time": False, "use_delta_time": True,},
#        {"time_embedding_size": 128, "use_raw_time": True, "use_delta_time": False,},
#        {"time_embedding_size": 128, "use_raw_time": False, "use_delta_time": True,},
    ]; name = "times"
    '''
    '''
    #args.train_epochs = 15
    opts = [
        {"latent_size": 64, "dec_recurrent_hidden_size": 64, "enc_hidden_size": 64,},
        {"latent_size": 32, "dec_recurrent_hidden_size": 64, "enc_hidden_size": 64,},
        {"latent_size": 16, "dec_recurrent_hidden_size": 64, "enc_hidden_size": 64,},
        {"latent_size": 8, "dec_recurrent_hidden_size": 64, "enc_hidden_size": 64,},
        {"latent_size": 64, "dec_recurrent_hidden_size": 32, "enc_hidden_size": 32,},
        {"latent_size": 32, "dec_recurrent_hidden_size": 32, "enc_hidden_size": 32,},
        {"latent_size": 16, "dec_recurrent_hidden_size": 32, "enc_hidden_size": 32,},
        {"latent_size": 8, "dec_recurrent_hidden_size": 32, "enc_hidden_size": 32,},
        {"latent_size": 64, "dec_recurrent_hidden_size": 16, "enc_hidden_size": 16,},
        {"latent_size": 32, "dec_recurrent_hidden_size": 16, "enc_hidden_size": 16,},
        {"latent_size": 16, "dec_recurrent_hidden_size": 16, "enc_hidden_size": 16,},
        {"latent_size": 8, "dec_recurrent_hidden_size": 16, "enc_hidden_size": 16,},
    ]; name = "latent"
    '''

    #{seq_len: {"hidden_sizes":[], "num_layers":[], "bidir":[], "train_samples":[], "scores":[]} for seq_len in eval_seq_lengths}
    try:
        #overall_results = pickle.load(open("{}/{}_search_results.pickle".format(base_path, name), "rb"))
        overall_results = pickle.load(open("{}/data_ablation/{}_search_results.pickle".format(base_path, name), "rb"))
    except:
        overall_results = {"options": [], "results": []}
    print(overall_results["options"])
    for i, options in enumerate(opts):#hs, nl, bi, ts in itertools.product(hidden_sizes, num_layers, bidirectional, train_samples):
        #args.train_data_percentage = pct
        def dict_match(a,b):
            a_k = set(a.keys())
            b_k = set(b.keys())
            if len(a_k) != len(b_k) or len(a_k) != len(a_k.intersection(b_k)) or len(b_k) != len(b_k.intersection(a_k)):
                return False
            for k in a_k:
                if a[k] != b[k]:
                    return False
            return True
        matches = [dict_match(options, cand_options) for cand_options in overall_results["options"]]
        if any(matches):
            print("Skipping {} due to already being present in results.".format(options))
            continue

        #if "train_data_percentage" in options:
        #    args.train_epochs = math.floor(orig_epochs / max(options["train_data_percentage"], 0.2))
        #    args.valid_epochs = max(args.valid_epochs-1, 1)
        #    args.save_epochs = max(args.save_epochs-1, 1)

        # ext = "_".join(["{}_{}".format(k,v) for k,v in options.items()])
        ext = "data_ablation"
        args.checkpoint_path = "{}/{}/".format(base_path.rstrip("/"), ext)
        # if os.path.exists(args.checkpoint_path):
        #     args.batch_size = orig_batch_size // 8
        # else:
        #     args.batch_size = orig_batch_size
        
        #args.time_embedding_size = options["time_embedding_size"]
        #args.use_raw_time = options["use_raw_time"]
        #args.use_delta_time = options["use_delta_time"]
        for k,v in options.items():
            args.__dict__[k] = v
        print("Starting run with options: {}".format(options))
        #args.recurrent_layers = nl
        #args.recurrent_bidirectional = bi
        #args.train_samples = ts

        #args.checkpoint_path = "{}/model_{}_{}_{}_{}.pt".format(
        #    base_path,
        #    hs,
        #    nl,
        #    "bidir" if bi else "forward",
        #    ts,
        #)

        while True:
            try:
                results = single_run(args)
                overall_results["results"].append(results)
                overall_results["options"].append(options)
                break
            except Exception as inst:
                print("Unexpected error:", inst)
                import traceback
                traceback.print_exc()
                time.sleep(60)
                pass
            #overall_results["results"].append("N/A")
        #for seq_len, score in eval_results.items():
        #    overall_results[seq_len]["hidden_sizes"] = hs
        #    overall_results[seq_len]["num_layers"] = nl
        #    overall_results[seq_len]["bidir"] = bi
        #    overall_results[seq_len]["train_samples"] = ts
        #    overall_results[seq_len]["scores"] = score 
        
        #pickle.dump(overall_results, open("{}/{}_search_results.pickle".format(base_path, name), "wb"))
        pickle.dump(overall_results, open("{}/data_ablation/{}_search_results.pickle".format(base_path, name), "wb"))

        time.sleep(10)
