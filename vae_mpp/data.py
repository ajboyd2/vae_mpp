import numpy as np
import pickle
import json
import os
import random

from collections import defaultdict
from parse import findall

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


PADDING_VALUES = {
    "ref_times": 1000.0,  # cannot be float('inf'), should be replaced with T + epsilon
    "ref_marks": 0,
    "tgt_times": 1000.0,  # cannot be float('inf'), should be replaced with T + epsilon
    "tgt_marks": 0,
    "ref_times_backwards": 1000.0,  # cannot be float('inf'), should be replaced with T + epsilon
    "ref_marks_backwards": 0,
    "padding_mask": 0,
}

def _ld_to_dl(ld, padded_size):
    """Converts list of dictionaries into dictionary of padded lists"""
    dl = defaultdict(list)
    for d in ld:
        for key, val in d.items():
            if key in PADDING_VALUES:
                new_val = F.pad(val, (0, padded_size - val.shape[-1]), value=PADDING_VALUES[key])
            else:
                new_val = val
            dl[key].append(new_val)
    return dl

def pad_and_combine_instances(batch):
    """
    A collate function for padding and combining instance dictionaries.
    """
    batch_size = len(batch)
    max_seq_len = max(max(len(ex["ref_times"]) for ex in batch), max(len(ex["tgt_times"]) for ex in batch))

    out_dict = _ld_to_dl(batch, max_seq_len)

    return {k: torch.stack(v, dim=0) for k,v in out_dict.items()}  # dim=0 means batch is the first dimension


class PointPatternDataset(Dataset):
    def __init__(
        self,
        file_path,
        args,
    ):
        """
        Loads text file containing realizations of point processes.
        Each line in the dataset corresponds to one realization.
        Each line will contain a comma-delineated sequence of "(t,k)"
        where "t" is the absolute time of the event and "k" is the associated mark.
        As of writing this, "t" should be a floating point number, and "k" should be a non-negative integer.
        The max value of "k" seen in the dataset determines the vocabulary size.
        """
        if len(file_path) == 1 and os.path.isdir(file_path[0]):
            file_path = [file_path[0].rstrip("/") + "/" + fp for fp in os.listdir(file_path[0])]
            file_path = sorted(file_path)
            print(file_path)

        self.user_mapping = {}
        self.user_id = {}
        if isinstance(file_path, list):
            self._instances = []
            self.vocab_size = 0
            for fp in file_path:
                instances, vocab_size = self.read_instances(fp)
                self._instances.extend(instances)
                self.vocab_size = max(self.vocab_size, vocab_size)
        else:
            self._instances, self.vocab_size = self.read_instances(file_path)

        self.same_tgt_and_ref = args.same_tgt_and_ref

    def __getitem__(self, idx):
        tgt_instance = self._instances[idx]

        tgt_times, tgt_marks = tgt_instance["times"], tgt_instance["marks"]

        if self.same_tgt_and_ref:
            ref_times, ref_marks = tgt_times, tgt_marks
        else:
            ref_instance = self._instances[random.choice(self.user_mapping[tgt_instance["user"]])]
            ref_times, ref_marks = ref_instance["times"], ref_instance["marks"]

        item = {
            'ref_times': torch.FloatTensor(ref_times),
            'ref_marks': torch.LongTensor(ref_marks), 
            'ref_times_backwards': torch.FloatTensor(np.ascontiguousarray(ref_times[::-1])), 
            'ref_marks_backwards': torch.LongTensor(np.ascontiguousarray(ref_marks[::-1])),
            'tgt_times': torch.FloatTensor(tgt_times),
            'tgt_marks': torch.LongTensor(tgt_marks),
            'padding_mask': torch.ones(len(tgt_marks), dtype=torch.uint8),
            'context_lengths': torch.LongTensor([len(ref_times) - 1]),  # these will be used for indexing later, hence the subtracting 1
            'T': torch.FloatTensor([tgt_instance["T"]]),
        }

        if "pp_obj_id" in tgt_instance:
            item["pp_id"] = torch.LongTensor([tgt_instance["pp_obj_id"]])

        if "user" in tgt_instance:
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
        vocab_size = max(max(instance["marks"]) for instance in instances) + 1

        for i, item in enumerate(instances):
            if "user" in item and (item["user"] not in self.user_mapping):
                self.user_mapping[item["user"]] = [i]
            elif "user" in item:
                self.user_mapping[item["user"]].append(i)

        return instances, vocab_size
