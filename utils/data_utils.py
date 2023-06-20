# -*- coding: utf-8 -*-
# data_utils
import json
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any
import random
import copy
from dataclasses import dataclass
from transformers import AutoTokenizer
from itertools import combinations
@dataclass
class ClassificationDataset(Dataset):
    data:List[dict]
    tokenizer:AutoTokenizer
    max_length:Optional[int]=None
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        inputs = []
        labels = []
        for b in batch:
            inputs.append('premise: '+b['premise']+'hypothesis: '+b['hypothesis'])
            if b.get('label') is not None:
                labels.append(b['label'])            
        if self.max_length is None:
            inputs = self.tokenizer(inputs, padding='longest',return_tensors = 'pt')
        else:
            inputs = self.tokenizer(inputs, padding=True, truncation=True, max_length=self.max_length, return_tensors = 'pt')
        if labels:
            inputs.data['labels']=torch.tensor(labels)
        return inputs
