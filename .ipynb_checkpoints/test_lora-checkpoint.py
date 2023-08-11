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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import argparse
from utils.data_utils import *
from utils.distributed_utils import *
from utils.utils import *
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from peft import PeftModel, PeftConfig

def get_args():
    # parser
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--test_data', type=str, help = 'test_data 위치')
    parser.add_argument('--output_dir', type=str, help = 'output 위치')
    parser.add_argument('--check_point_dir', type=str)
    parser.add_argument('--model_path', type=str)
    
    # 데이터 관련
    parser.add_argument('--max_length',type= int, default = 512)
    parser.add_argument('--batch_size', default = 8, type=int)
    parser.add_argument('--answer_max_length', default = 20, type=int)
    
    
    # TODO
    ## distributed 관련
    parser.add_argument('--local_rank', type=int, default = -1)
    parser.add_argument('--distributed', type=str2bool, default = False)
    args = parser.parse_args()
    return args

def get_tokenizer_and_model(args):
    if any(k in args.ptm_path for k in ("gpt", "opt", "bloom", "Polyglot")):
        padding_side = "left"
    else:
        padding_side = "right"
    tokenizer = AutoTokenizer.from_pretrained(args.ptm_path, padding_side=padding_side)
    
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    config = PeftConfig.from_pretrained(args.model_path)
    
    if 't5' in args.ptm_path:
        model =  AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, num_labels=args.n_labels, torch_dtype="auto", low_cpu_mem_usage=True, pad_token_id=tokenizer.pad_token_id)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=args.n_labels, torch_dtype="auto", low_cpu_mem_usage=True, pad_token_id=tokenizer.pad_token_id)
    model = PeftModel.from_pretrained(model, args.model_path)
    return tokenizer, model 

def synchronize(args, check_point_args):
    args.n_labels = check_point_args['n_labels']
    args.ptm_path = check_point_args['ptm_path']
    return args

# evaluation
def evaluation(args, model, tokenizer, eval_dataloader):
    total_loss = 0.
    model.eval()
    predicts = []
    actuals = []
    with torch.no_grad():
        for data in tqdm(eval_dataloader, desc = 'evaluate', disable =  args.local_rank not in [-1,0]):
            data = {i:j.cuda() for i,j in data.items() if i!='token_type_ids'} 
            if 't5' in args.ptm_path:
                labels = copy.deepcopy(data['labels'])
                data['labels'][data['labels']==tokenizer.pad_token_id]=-100 # 굉장히 중요.
            output = model.forward(**data)
            loss = output.loss
            total_loss+=loss.item()
            if 't5' in args.ptm_path:
                model_to_generate = model.module if hasattr(model,'module') else model
                data.pop('labels')
                outputs = model_to_generate.generate(
                **data,
                pad_token_id = tokenizer.pad_token_id,
                decoder_start_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.eos_token_id,
                eos_token_id = tokenizer.eos_token_id,
                early_stopping = True,
                do_sample = False,
                num_beams = 20,
                max_length = args.answer_max_length,
                )
                predict = tokenizer.batch_decode(outputs, skip_special_tokens = True)
                actual = tokenizer.batch_decode(labels.cpu(), skip_special_tokens = True)
            else:
                predict = output.logits.argmax(dim=-1).cpu().tolist()
                actual = data['labels'].cpu().tolist()
            predicts.extend(predict)
            actuals.extend(actual)
    acc = accuracy_score(actuals, predicts)
    f1 = f1_score(actuals, predicts, average='weighted')
    return dict(loss=total_loss/len(eval_dataloader), acc=acc, f1=f1), predicts, actuals

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
    if 't5' in args.ptm_path: 
        test_dataset = T5NLIDataset(test_data, tokenizer, args.max_length)
    else:
        test_dataset = NLIDataset(test_data, tokenizer, args.max_length)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, sampler = test_sampler, collate_fn = test_dataset.collate_fn)
    
    scores, predicts, actuals = evaluation(args, model, tokenizer, test_dataloader)
    print(scores)
    if 't5' in args.ptm_path: 
        predicts = [TEXT2LABEL[i] for i in predicts]
        actuals = [TEXT2LABEL[i] for i in actuals]
    print(confusion_matrix(actuals, predicts))
    ###########################################################################################
    
