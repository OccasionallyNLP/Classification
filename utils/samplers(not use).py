import math
from typing import TypeVar, Optional, Iterator
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import *

class DistributedWeightedRandomSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, replacement: bool = False) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.replacement = replacement

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        # weights
        n_class = Counter([i['label'] for i in self.dataset]) 
        class_weight = {i:1/j for i,j in n_class.items()}
        weights = torch.DoubleTensor([class_weight[i['label']] for i in self.dataset])
        rand_tensor = torch.multinomial(weights, self.num_samples, self.replacement, generator=None)
        yield from iter(rand_tensor.tolist())
