# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Dict, Optional
from transformers import PreTrainedModel, T5EncoderModel
from dataclasses import dataclass

# point wise - classification
class ClassificationModel(PreTrainedModel):
    def __init__(self, config, pool, model_class, n_labels):
        super().__init__(config)
        self.pool = pool
        self.pretrained_model = model_class(config) 
        if model_class == T5EncoderModel:
            self.fc = nn.Linear(config.d_model, n_labels)
        else:
            self.fc = nn.Linear(config.hidden_size, n_labels) 
        
    def init_pretrained_model(self, state_dict):
        self.pretrained_model.load_state_dict(state_dict) 
    
    def forward(self, input_ids, attention_mask, **kwargs):
        output = self.pretrained_model(input_ids, attention_mask) 
        if self.pool=='cls':
            out = output['pooler_output'] # bs, dim
        elif self.pool == 'mean':
            out = output['last_hidden_state'].masked_fill(attention_mask.unsqueeze(2).repeat(1,1,self.config.hidden_size)==0,0) # bs, seq_len, dim
            out = out.sum(dim=1) # bs, dim
            s = attention_mask.sum(-1, keepdim=True) # bs, dim
            out = out/(s)
            
        scores = self.fc(out) # bs, n_labels
        return dict(score = scores)
    
# point wise - classification
class MTClassificationModel(PreTrainedModel):
    def __init__(self, config, pool, model_class, n_labels:list):
        # n_labels - 각 category 마다  n_labels -> 
        super().__init__(config)
        self.pool = pool
        self.pretrained_model = model_class(config) 
        
        if model_class == T5EncoderModel:
            self.fcs = nn.ModuleList([nn.Linear(config.d_model, i) for i in n_labels])
        else:
            self.fcs = nn.ModuleList([nn.Linear(config.hidden_size, i) for i in n_labels])
        
    def init_pretrained_model(self, state_dict):
        self.pretrained_model.load_state_dict(state_dict) 
    
    def forward(self, input_ids, attention_mask, **kwargs):
        output = self.pretrained_model(input_ids, attention_mask) 
        if self.pool=='cls':
            out = output['pooler_output'] # bs, dim
        elif self.pool == 'mean':
            out = output['last_hidden_state'].masked_fill(attention_mask.unsqueeze(2).repeat(1,1,self.config.hidden_size)==0,0) # bs, seq_len, dim
            out = out.sum(dim=1) # bs, dim
            s = attention_mask.sum(-1, keepdim=True) # bs, dim
            out = out/(s)
        
        scores = {'score_%d'%k:i(out) for k,i in enumerate(self.fcs)} # bs, n_labels
        return scores
