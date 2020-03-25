import numpy as np
import pickle
import json
import os
import random
import math

from collections import defaultdict
from parse import findall

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


PADDING_VALUES = {
    "ref_times": None, # 1000.0 # cannot be float('inf'), should be replaced with T + epsilon
    "ref_marks": 0,
    "tgt_times": None, # 1000.0,  # cannot be float('inf'), should be replaced with T + epsilon
    "tgt_marks": 0,
    "ref_times_backwards": None, # 1000.0,  # cannot be float('inf'), should be replaced with T + epsilon
    "ref_marks_backwards": 0,
    "padding_mask": 0,
    "augment_mask": [0.0],
}

def _ld_to_dl(ld, padded_size, def_pad_value):
    """Converts list of dictionaries into dictionary of padded lists"""
    dl = defaultdict(list)
    for d in ld:
        for key, val in d.items():
            if key in PADDING_VALUES:
                pad_val = PADDING_VALUES[key]
                if pad_val is None:
                    pad_val = def_pad_value
                if isinstance(pad_val, list):
                    new_val = F.pad(val, (0, 0, 0, padded_size - val.shape[-2]), value=pad_val[0])
                else:
                    new_val = F.pad(val, (0, padded_size - val.shape[-1]), value=pad_val)
            else:
                new_val = val
            dl[key].append(new_val)
    return dl

def pad_and_combine_instances(batch, def_pad_value):
    """
    A collate function for padding and combining instance dictionaries.
    """
    batch_size = len(batch)
    max_seq_len = max(max(len(ex["ref_times"]) for ex in batch), max(len(ex["tgt_times"]) for ex in batch))

    out_dict = _ld_to_dl(batch, max_seq_len, def_pad_value)

    return {k: torch.stack(v, dim=0) for k,v in out_dict.items()}  # dim=0 means batch is the first dimension


class PointPatternDataset(Dataset):
    def __init__(
        self,
        file_path,
        args,
        keep_pct,
        set_dominating_rate,
    ):
        """
        Loads text file containing realizations of point processes.
        Each line in the dataset corresponds to one realization.
        Each line will contain a comma-delineated sequence of "(t,k)"
        where "t" is the absolute time of the event and "k" is the associated mark.
        As of writing this, "t" should be a floating point number, and "k" should be a non-negative integer.
        The max value of "k" seen in the dataset determines the vocabulary size.
        """
        self.keep_pct = keep_pct
        self.max_channels = args.num_channels

        if len(file_path) == 1 and os.path.isdir(file_path[0]):
            file_path = [file_path[0].rstrip("/") + "/" + fp for fp in os.listdir(file_path[0])]
            file_path = sorted(file_path)
            print(file_path)

        self.user_mapping = {}
        self.user_id = {}
        if isinstance(file_path, list):
            self.is_valid = any(["valid" in fp for fp in file_path])
            self._instances = []
            self.vocab_size = 0
            for fp in file_path:
                instances, vocab_size = self.read_instances(fp)
                self._instances.extend(instances)
                self.vocab_size = max(self.vocab_size, vocab_size)
        else:
            self.is_valid = "valid" in file_path
            self._instances, self.vocab_size = self.read_instances(file_path)

        self.same_tgt_and_ref = args.same_tgt_and_ref

        self.do_augment = args.augment_loss_coef > 0
        self.surprise_augment = args.augment_loss_surprise and self.do_augment

        # find a dominating rate for the dataset for the purposes of sampling
        if set_dominating_rate:
            max_rate = 0
            avg_rate = 0
            for instance in self._instances:
                avg_rate += len(instance["times"]) / instance["T"]
                for i in range(3, len(instance["times"])):
                    diff = (instance["times"][i] - instance["times"][i-3])
                    if diff > 0:
                        max_rate = max(max_rate, 4 / diff) 
            avg_rate /= len(self._instances)
            args.dominating_rate = 50 * avg_rate 

            print("For Data Loaded, average rate {}, max rate {}, dominating rate {}".format(avg_rate, max_rate, args.dominating_rate))

        max_period = 0
        for instance in self._instances:
            max_period = max(max_period, instance["T"])
        self.max_period = max_period

    def __getitem__(self, idx):
        tgt_instance = self._instances[idx]

        tgt_times, tgt_marks = tgt_instance["times"], tgt_instance["marks"]

        if self.same_tgt_and_ref:
            ref_times, ref_marks = tgt_times, tgt_marks
        else:
            if self.is_valid:
                ref_instance = self._instances[random.choice([ref_idx for ref_idx in self.user_mapping[tgt_instance["user"]] if ref_idx != idx])]
            else:
                ref_instance = self._instances[random.choice(self.user_mapping[tgt_instance["user"]])]
            ref_times, ref_marks = ref_instance["times"], ref_instance["marks"]

        tgt_marks = torch.LongTensor(tgt_marks)

        if self.do_augment:
            expanded_marks = torch.nn.functional.one_hot(tgt_marks, num_classes=self.max_channels)
            if self.surprise_augment:
                expanded_marks = expanded_marks.cumsum(dim=0)
            else:
                expanded_marks = expanded_marks.sum(dim=0).expand_as(expanded_marks)
            augment_mask = torch.where( 
                expanded_marks == 0, 
                torch.ones_like(expanded_marks, dtype=torch.float32), 
                torch.zeros_like(expanded_marks, dtype=torch.float32),
            )
            augment_mask_count = augment_mask.sum(dim=-1).unsqueeze(-1)
            if (augment_mask_count == 0).any().item():
                print("ZERO COUNT")
                print(augment_mask)
                input()
            augment_mask = augment_mask / augment_mask_count
        else:
            augment_mask = torch.zeros(tgt_marks.shape[0], self.max_channels)

        item = {
            'ref_times': torch.FloatTensor(ref_times),
            'ref_marks': torch.LongTensor(ref_marks), 
            'ref_times_backwards': torch.FloatTensor(np.ascontiguousarray(ref_times[::-1])), 
            'ref_marks_backwards': torch.LongTensor(np.ascontiguousarray(ref_marks[::-1])),
            'tgt_times': torch.FloatTensor(tgt_times),
            'tgt_marks': tgt_marks,  # made into a tensor for loss augmentation
            'padding_mask': torch.ones(len(tgt_marks), dtype=torch.uint8),
            'context_lengths': torch.LongTensor([len(ref_times) - 1]),  # these will be used for indexing later, hence the subtracting 1
            'T': torch.FloatTensor([tgt_instance["T"]]),
            'augment_mask': augment_mask,
        }

        if "pp_obj_id" in tgt_instance:
            item["pp_id"] = torch.LongTensor([tgt_instance["pp_obj_id"]])

        if "user" in tgt_instance:
            #print("USER", tgt_instance["user"])
            if tgt_instance["user"] not in self.user_id:
                self.user_id[tgt_instance["user"]] = len(self.user_id)
            item["pp_id"] = torch.LongTensor([self.user_id[tgt_instance["user"]]])

        return item

    def __len__(self):
        return len(self._instances)

    def get_max_T(self):
        return max(item["T"] for item in self._instances)

    def read_instances(self, file_path):
        """Load PointProcessDataset from a file"""

        if ".pickle" in file_path:
            with open(file_path, "rb") as f:
                collection = pickle.load(f)
            instances = collection["sequences"]
            for instance in instances:
                if "T" not in instance:
                    instance["T"] = 50.0
        elif ".json" in file_path:
            instances = []
            with open(file_path, 'r') as f:
                for line in f:
                    instances.append(json.loads(line))
        else:
            with open(file_path, 'r') as f:
                instances = []
                for line in f:
                    items = [(float(r.fixed[0]), int(r.fixed[1])) for r in findall("({},{})", line.strip())]
                    times, marks = zip(*items)
                    instances.append({
                        "user": file_path,
                        "T": 50.0,
                        "times": times,
                        "marks": marks
                    })
        for i in range(len(instances)):
            instance = instances[i]
            keep_idx = [j for j,m in enumerate(instance["marks"]) if m < self.max_channels]
            instance["times"] = [instance["times"][j] for j in keep_idx]
            instance["marks"] = [instance["marks"][j] for j in keep_idx]
        instances = [instance for instance in instances if len(instance["times"]) > 0]

        vocab_size = max(max(instance["marks"]) for instance in instances) + 1

        if self.is_valid:
            user_counts = {}
            for instance in instances:
                user = instance["user"]
                if user in user_counts:
                    user_counts[user] += 1
                else:
                    user_counts[user] = 1
            user_counts = {k for k,v in user_counts.items() if v > 1}
            instances = [instance for instance in instances if instance["user"] in user_counts]

        if self.keep_pct < 1:
            old_len = len(instances)
            users = list(set(instance["user"] for instance in instances))
            indices = list(range(len(users)))
            random.shuffle(indices)
            indices = sorted(indices[:math.floor(len(users) * self.keep_pct)])
            users = set(users[idx] for idx in indices)
            instances = [instance for instance in instances if instance['user'] in users]
            print("Before filtering: {} | After filtering: {} | Prop: {} | Goal: {}".format(old_len, len(instances), len(instances) / old_len, self.keep_pct))

        for i in range(len(instances)):
            if instances[i]["times"][0] == 0.0:
                instances[i]["times"][0] += 1e-8
                assert((len(instances[i]["times"]) == 1) or (instances[i]["times"][0] < instances[i]["times"][1]))

        for i, item in enumerate(instances):
            if "user" in item and (item["user"] not in self.user_mapping):
                self.user_mapping[item["user"]] = [i]
            elif "user" in item:
                self.user_mapping[item["user"]].append(i)

        med_len = sorted([len(item["times"]) for item in instances])[len(instances)//2]
        avg_len = sum(len(item["times"]) for item in instances) / len(instances)
        med_su = sorted([len(v) for k,v in self.user_mapping.items()])[len(self.user_mapping) // 2]
        print("SEQS {} | USERS {} | Med S/U {} | Avg S/U {} | Med SEQ LEN {} | Avg SEQ LEN {}".format(
            len(instances), len(self.user_mapping), med_su, len(instances)/len(self.user_mapping), med_len, avg_len,
        ))

        return instances, vocab_size
