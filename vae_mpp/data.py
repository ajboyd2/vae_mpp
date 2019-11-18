import numpy as np
import pickle
import json
import os

from collections import defaultdict
from parse import findall

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


PADDING_VALUES = {
    "times": 1000.0,  # cannot be float('inf'), should be replaced with T + epsilon
    "marks": 0,
    "times_backwards": 1000.0,  # cannot be float('inf'), should be replaced with T + epsilon
    "marks_backwards": 0,
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
    max_seq_len = max(len(ex["times"]) for ex in batch)

    out_dict = _ld_to_dl(batch, max_seq_len)

    return {k: torch.stack(v, dim=0) for k,v in out_dict.items()}  # dim=0 means batch is the first dimension


class PointPatternDataset(Dataset):
    def __init__(self,
                 file_path):
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
            print(file_path)

        if isinstance(file_path, list):
            self._instances = []
            self.vocab_size = 0
            for fp in file_path:
                instances, vocab_size = self.read_instances(fp)
                self._instances.extend(instances)
                self.vocab_size = max(self.vocab_size, vocab_size)
        else:
            self._instances, self.vocab_size = self.read_instances(file_path)
        self.user_mapping = {}

    def __getitem__(self, idx):
        instance =  self._instances[idx]

        item = {
            'times': torch.FloatTensor(instance["times"]),
            'marks': torch.LongTensor(instance["marks"]), 
            'times_backwards': torch.FloatTensor(np.ascontiguousarray(instance["times"][::-1])), 
            'marks_backwards': torch.LongTensor(np.ascontiguousarray(instance["marks"][::-1])),
            'padding_mask': torch.ones(len(instance["marks"]), dtype=torch.uint8),
            'context_lengths': torch.LongTensor([len(instance["times"]) - 1]),  # these will be used for indexing later, hence the subtracting 1
            'T': torch.FloatTensor([instance["T"]]),
        }

        if "pp_obj_id" in instance:
            item["pp_id"] = torch.LongTensor([instance["pp_obj_id"]])

        if "user" in instance:
            if instance["user"] not in self.user_mapping:
                self.user_mapping[instance["user"]] = len(self.user_mapping)
            item["pp_id"] = torch.LongTensor([self.user_mapping[instance["user"]]])

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
                        "T": 50.0,
                        "times": times,
                        "marks": marks
                    })
        vocab_size = max(max(instance["marks"]) for instance in instances) + 1

        return instances, vocab_size
