# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Dict, Optional
from transformers import PreTrainedModel, T5EncoderModel
from dataclasses import dataclass

def mean_pool(output, pool_type):
    if pool_type=='cls':
        out = output['pooler_output'] # bs, dim
    elif pool == 'mean':
        out = output['last_hidden_state'].masked_fill(attention_mask.unsqueeze(2).repeat(1,1,self.config.hidden_size)==0,0) # bs, seq_len, dim
        out = out.sum(dim=1) # bs, dim
        s = attention_mask.sum(-1, keepdim=True) # bs, dim
        out = out/(s)
    return out