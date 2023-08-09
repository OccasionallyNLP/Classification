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
from transformers import AutoTokenizer, BertModel, RobertaModel,T5EncoderModel, AutoConfig, AutoModel
import argparse
from utils.data_utils import *
from utils.distributed_utils import *
from utils.utils import *
from model import *
from sklearn.metrics import accuracy_score, f1_score

from losses import *
from sklearn.metrics import accuracy_score, f1_score

def calc_loss(args, logits, labels, weights=None):
    # label에 따른 차이가 필요함.
    if weights is not None:
        weights = weights.to(labels)
    if args.n_labels != 1:
            # just for binary classification
        if args.focal_loss:
            loss_fn = FocalLoss(args.gamma, weights)
        else:
            loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        if args.focal_loss:
            loss_fn = BinaryFocalLoss(weights, args.gamma)
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        logits = logits.squeeze(1)
        labels = labels.float()
    loss = loss_fn(logits, labels)
    return loss

# evaluation
def evaluation(args, model, tokenizer, eval_dataloader):
    total_loss = 0.
    model.eval()
    predicts = []
    actuals = []
    logits = []
    with torch.no_grad():
        for data in tqdm(eval_dataloader, desc = 'evaluate', disable =  args.local_rank not in [-1,0]):
            data = {i:j.cuda() for i,j in data.items()}
            output = model.forward(**data)
<<<<<<< HEAD
            loss = calc_loss(args, output['score'], data['labels'], weights=args.weights)
=======
            loss = calc_loss(args, output['score'], data['labels'], weights=weights)
>>>>>>> ad30057295cd69f9a1138f6c8b7f6f52753d3a32
            total_loss+=loss.item()
            if args.n_labels == 1:
                predict = (torch.sigmoid(output['score'])>=0.5).squeeze(1).long().cpu().tolist()
            else:
                predict = output['score'].argmax(dim=-1).cpu().tolist()
            logits.append(output['score'].cpu().tolist())
            actual = data['labels'].cpu().tolist()
            predicts.extend(predict)
            actuals.extend(actual)
    acc = accuracy_score(actuals, predicts)
    f1 = f1_score(actuals, predicts, average='weighted')
    return dict(loss=total_loss/len(eval_dataloader), acc=acc, f1=f1), predicts, actuals, logits

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
    config = AutoConfig.from_pretrained(args.ptm_path)
    if 'bert' in args.ptm_path:
        model_class = BertModel
    elif 'roberta' in args.ptm_path:
        model_class = RobertaModel
    elif 'ulm' in args.ptm_path:
        model_class = T5EncoderModel
    model = ClassificationModel(config, args.pool, model_class, args.n_labels)
    if args.model_path is None:
        if 'ulm' in args.ptm_path:
            backbone_model = T5EncoderModel.from_pretrained(args.ptm_path)
        else:
            backbone_model = AutoModel.from_pretrained(args.ptm_path)
        model.init_pretrained_model(backbone_model.state_dict())
    else:
        model_state_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(model_state_dict)
    return tokenizer, model 

def synchronize(args, check_point_args):
    args.weight = torch.tensor(check_point_args['weights']) if check_point_args['weights'] is not None else check_point_args['weights'] 
    args.weighted_cross_entropy = check_point_args['weighted_loss']
    args.focal_loss = check_point_args['focal_loss']
    args.gamma = check_point_args['gamma']
    args.pool = check_point_args['pool']
    args.n_labels = check_point_args['n_labels']
    args.ptm_path = check_point_args['ptm_path']
    args.model_path = os.path.join(check_point_args['output_dir'],'best_model')
    return args

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
    test_dataset = ClassificationDataset(test_data, tokenizer, args.max_length)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, sampler = test_sampler, collate_fn = test_dataset.collate_fn)
    
    scores, predicts, actuals, logits = evaluation(args, model, tokenizer, test_dataloader)
    print(scores)
    ###########################################################################################
    
