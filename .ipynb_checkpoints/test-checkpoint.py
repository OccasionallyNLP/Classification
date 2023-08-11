# test T5
# -*- coding: utf-8 -*-
import os
import json
import pickle
import copy
from tqdm import tqdm
import time
import re
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
from utils.data_utils import *
from utils.distributed_utils import *
from utils.utils import *
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)

def get_args():
    # parser
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--test_data', type=str, help = 'test_data 위치')
    parser.add_argument('--output_dir', type=str, help = 'output 위치')
    parser.add_argument('--check_point_dir', type=str)
    
    # 데이터 관련
    parser.add_argument('--max_length',type= int, default = 512)
    parser.add_argument('--batch_size', default = 8, type=int)
    
    # TODO
    ## distributed 관련
    parser.add_argument('--local_rank', type=int, default = -1)
    parser.add_argument('--distributed', type=str2bool, default = False)
    args = parser.parse_args()
    return args

def get_tokenizer_and_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.ptm_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.ptm_path, num_labels=args.n_labels)
    if args.model_path is not None:
        model_state_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(model_state_dict)
    return tokenizer, model 

def synchronize(args, check_point_args):
    args.n_labels = check_point_args['n_labels']
    args.ptm_path = check_point_args['ptm_path']
    args.model_path = os.path.join(check_point_args['output_dir'],'best_model')
    return args

# evaluation
def evaluation(args, model, tokenizer, eval_dataloader):
    total_loss = 0.
    model.eval()
    predicts = []
    actuals = []
    with torch.no_grad():
        for data in tqdm(eval_dataloader, desc = 'evaluate', disable =  args.local_rank not in [-1,0]):
            data = {i:j.cuda() for i,j in data.items()}
            output = model.forward(**data)
            loss = output.loss
            total_loss+=loss.item()
            if args.n_labels == 1:
                predict = output.logits.squeeze(dim=-1).cpu().tolist()
            else:
                predict = output.logits.argmax(dim=-1).cpu().tolist()
            actual = data['labels'].cpu().tolist()
            predicts.extend(predict)
            actuals.extend(actual)
    if args.n_labels == 1:
        acc = mean_squared_error(actuals, predicts)
    else:
        acc = accuracy_score(actuals, predicts)
    return dict(loss=total_loss/len(eval_dataloader), acc=acc), predicts, actuals

if __name__=='__main__':
    args  = get_args()
    seed_everything(42)
    # sanity check
    os.makedirs(args.output_dir, exist_ok = True)
    with open(os.path.join(args.check_point_dir,'args.txt'), 'r') as f:
        check_point_args = json.load(f)    
    args = synchronize(args, check_point_args)
    
    ###########################################################################################
    # tokenizer, config, model
    ###########################################################################################
    tokenizer, model = get_tokenizer_and_model(args)
    ###########################################################################################
    # device
    ###########################################################################################
    # TODO
    if args.local_rank == -1:  # single-node multi-gpu (or cpu) mode
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
    # multi gpu
    else:
        device = torch.device(f'cuda:{args.local_rank}')
        torch.cuda.set_device(device) 
        model.to(device)
    ###########################################################################################
    
    ###########################################################################################
    # data
    ###########################################################################################
    test_data = load_jsonl(args.test_data)
    
    # just list wise when evaluates
    test_dataset = NLIDataset(test_data, tokenizer, args.max_length)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, sampler = test_sampler, collate_fn = test_dataset.collate_fn)
    
    scores, predicts, actuals = evaluation(args, model, tokenizer, test_dataloader)
    print(scores)
    print(confusion_matrix(actuals, predicts))
    ###########################################################################################
    
