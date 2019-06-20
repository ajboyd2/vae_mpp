from collections import defaultdict
import logging
from parser import findall

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

def pad_and_combine_instances(batch):
    """
    A collate function for padding and combining instance dictionaries.
    """
    batch_size = len(batch)
    max_field_lengths = defaultdict(int)
    for instance in batch:
        for field, tensor in instance.items():
            if len(tensor.size()) == 0:
                continue
            elif len(tensor.size()) == 1:
                max_field_lengths[field] = max(max_field_lengths[field], tensor.size()[0])
            elif len(tensor.size()) > 1:
                raise ValueError('Padding multi-dimensional tensors not supported')

    out_dict = {}
    for i, instance in enumerate(batch):
        for field, tensor in instance.items():
            if field not in out_dict:
                if field in max_field_lengths:
                    size = (batch_size, max_field_lengths[field])
                else:
                    size = (batch_size,)
                out_dict[field] = tensor.new_zeros(size)
            if field in max_field_lengths:
                out_dict[field][i, -tensor.size()[0]:] = tensor  # prepend padding for efficiency due to sorting
            else:
                out_dict[field][i] = tensor

    return out_dict


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
        self._instances, self.vocab_size = self.read_instances(file_path)

    def __getitem__(self, idx):
        instance =  self._instances[idx]
        return {
            'times': torch.FloatTensor(instance["times"]),
            'marks': torch.LongTensor(instance["marks"]),
            'padding_mask': torch.ones(len(instance["marks"]), dtype=torch.uint8)
        }

    def __len__(self):
        return len(self._instances)

    def read_instances(self, file_path):
        """Load PointProcessDataset from a file"""

        instances = []
        with open(file_path, 'r') as f:
            for line in f:
                items = [(float(r.fixed[0]), int(r.fixed[1])) for r in p.findall("({},{})", line.strip())]
                times, marks = zip(*items)
                instances.append({
                    "times": times,
                    "marks": marks
                })
        logger.debug('Max seq. len: %i', max(len(x) for x in instances))
        vocab_size = max(k for instance in instances for _, k in instance)

        return instances, vocab_size
