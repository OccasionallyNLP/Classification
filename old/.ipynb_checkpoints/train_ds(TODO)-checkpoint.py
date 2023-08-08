# -*- coding: utf-8 -*-
import os
import json
import pickle
import copy
from tqdm import tqdm
import logging
import time
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_constant_schedule_with_warmup
import argparse
from utils.data_utils import *
from utils.distributed_utils import *
from utils.utils import *
from utils.metrics import *
from model import *
import deepspeed 
from transformers.deepspeed import HfDeepSpeedConfig

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
            predict = output.logits.argmax(dim=-1).cpu().tolist()
            actual = data['labels'].cpu().tolist()
            predicts.extend(predict)
            actuals.extend(actual)
    acc = accuracy_score(actuals, predicts)
    f1 = f1_score(actuals, predicts, average='weighted')
    return dict(loss=total_loss/len(eval_dataloader), acc=acc, f1=f1)

def get_scores(local_rank, scores, distributed:bool):
    output = {}
    if distributed:
        for i,j in scores.items():
            tmp = [j.item() for j in get_global(local_rank, torch.tensor([j]).cuda())]
            output[i] = np.round(sum(tmp)/len(tmp),4)
    else:
        for i,j in scores.items():
            output[i] = np.round(j,4)
    return output

def get_args():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, help = 'test_name')
    parser.add_argument('--output_dir', type=str, help = 'output 위치')
    # data
    parser.add_argument('--train_data', type=str, help = 'train_data 위치')
    parser.add_argument('--val_data', type=str, help='val data 위치')
    
    # logging 관련
    parser.add_argument('--logging_term', type=int, default = 100)
   
    # 학습 관련
    parser.add_argument('--epochs', default = 10, type=int)
    parser.add_argument('--eval_epoch', type = int, default = 1, help = 'term of evaluation')
    parser.add_argument('--batch_size', default = 8, type=int)
    parser.add_argument('--lr', type=float, default = 5e-5)
    parser.add_argument('--warmup', type=float, default = 1000)
    parser.add_argument('--decay', type=float, default = 0.05)
    
    # 경량화
    parser.add_argument('--fp16', type=str2bool, default = False)
    parser.add_argument('--fp16_model', type=str2bool, default = False)
    
    
    parser.add_argument('--accumulation_steps', type=int, default = 1) # 221124 추가
    parser.add_argument('--weighted_sampling', type=str2bool, default = False) # 221124 추가
    # PTM model
    parser.add_argument('--ptm_path', type=str)
    parser.add_argument('--model_path', type=str)
    
    # model input
    parser.add_argument('--max_length', type=int)
    
    # distributed 관련
    parser.add_argument('--local_rank', type=int, default = -1)
    parser.add_argument('--distributed', type=str2bool, default = False)
    parser.add_argument('--early_stop', type=str2bool, default = True) # XXX220919
    parser.add_argument('--patience', type=int, default = 3)
    
    # deepspeed
    # deepspeed
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def synchronize_config_with_deepspeed(args, dataset):
    ds_config = json.load(open(args.deepspeed_config,'r',encoding='utf-8'))
    # batch size
    args.batch_size = ds_config['train_micro_batch_size_per_gpu']
    total_step = int(len(dataset)/args.batch_size)
    warmup = total_step * args.warmup
    ds_config['scheduler']['params']['warmup_num_steps']=warmup
    args.fp16 = ds_config['fp16']['enabled']
    args.logging_term = ds_config['steps_per_print']
    return ds_config

def update_lr(args, global_step, optimizer, scheduler_fn):
    lr_this_step = args.lr * scheduler_fn(global_step)
    for param_group in optimizer.param_groups:
        param_group['lr']=lr_this_step
    return lr_this_step

def train():
    if args.local_rank in [-1,0]:
        early_stop = EarlyStopping(args.patience, args.output_dir, max = False, min_difference=1e-5)
    flag_tensor = torch.zeros(1).cuda()
    scheduler_fn = lambda step : min(1/model.lr_scheduler.warmup_num_steps*step, 1.)
    ########################################################################################
    # train
    ########################################################################################
    last_lr = 0
    global_step = 0
    for epoch in range(1, args.epochs+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        Loss = 0.
        step = 0
        iter_bar = tqdm(train_dataloader, desc='step', disable=args.local_rank not in [-1,0])
        #train
        for data in iter_bar:
            optimizer.zero_grad()            
            data = {i:j.cuda() for i,j in data.items()}
            loss = model.forward(**data)['loss']
            
            if args.distributed:
                loss = loss.mean()
            
            if args.accumulation_steps > 1 :
                loss = loss / args.accumulation_steps
            
            # backward loss
            model.backward(loss)
            
            if step%args.accumulation_steps==0 or (
                    len(train_dataloader) <= args.accumulation_steps
                    and (step) == len(train_dataloader)
            ):
                model.step()
                if args.fp16:
                    last_lr = update_lr(args, global_step, optimizer, scheduler_fn)
                else:
                    last_lr = model.lr_scheduler.get_last_lr()[0]
            Loss+=loss.item()
            step+=1
            global_step+=1
            iter_bar.set_postfix({'epoch':epoch, 'global_step':global_step, 'lr':f"{last_lr:.5f}",'total_loss':f'{Loss/step:.5f}'}) # 감소한다는 것을 확인하는 것임.
            if global_step%args.logging_term == 0:
                if args.local_rank in [-1,0]:
                    logger1.info(iter_bar)
                    logger2.info(iter_bar)
            
        # epoch 당 기록.
        if args.local_rank in [-1,0]:
            logger1.info(iter_bar)
            logger2.info(iter_bar)
        ########################################################################################
        # evaluation
        ###################################################################################################
        if args.eval_epoch!=0 and epoch%args.eval_epoch==0:
            # validation
            val_scores_, _ = evaluation(args, model, tokenizer, val_dataloader)
            val_scores = get_scores(args.local_rank, val_scores_, args.distributed)            
            
            if args.local_rank in [-1,0]:
                logger1.info(f'Val ---- epoch : {epoch} ----- scores:{val_scores}')
                logger2.info(f'Val ---- epoch : {epoch} ----- scores:{val_scores}')
                model_to_save = model.module if hasattr(model,'module') else model
                early_stop.check(model_to_save, val_scores['Loss'])  
                if early_stop.timetobreak:
                    flag_tensor += 1
            if args.distributed:
                torch.distributed.broadcast(flag_tensor, 0) 
                torch.distributed.barrier()
        ###################################################################################################
        if args.early_stop:    
            if flag_tensor:
                if args.local_rank in [-1,0]:
                    logger1.info('early stop')
                    logger2.info('early stop')
                break
    # 저장시 - gpu 0번 것만 저장 - barrier 필수
    if args.local_rank in [-1,0]:
        torch.save(early_stop.best_model, os.path.join(early_stop.save_dir,'best_model'))
        logger1.info('train_end')
        logger2.info('train end')

def get_tokenizer_and_model(args):
    if any(k in args.ptm_path for k in ("gpt", "opt", "bloom", "Polyglot")):
        padding_side = "left"
    else:
        padding_side = "right"
    tokenizer = AutoTokenizer.from_pretrained(args.ptm_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForSequenceClassification.from_pretrained(args.ptm_path, num_labels=args.n_labels, torch_dtype="auto", low_cpu_mem_usage=True)
    
    # kogpt
#     tokenizer = AutoTokenizer.from_pretrained(
#   'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
#   bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]',
# padding_side=padding_side)
#     model = AutoModelForSequenceClassification.from_pretrained(
#   'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
#   pad_token_id=tokenizer.eos_token_id,
#   torch_dtype='auto', low_cpu_mem_usage=True, num_labels=args.n_labels
# )
    
    if args.model_path is not None:
        model_state_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(model_state_dict)
    
    # peft
    if 'bert' in args.ptm_path or 'roberta' in args.ptm_path:
        peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=args.r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    elif 't5' in args.ptm_path:
        pass
    else:
        peft_config = LoraConfig(task_type="CAUSAL_LM", inference_mode=False, r=args.r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
        
    model = get_peft_model(model, peft_config)
    print('model trainable parameters')
    model.print_trainable_parameters()
    return tokenizer, model 

def load_datasets(args, tokenizer):
    # LOAD DATASETS
    train_data = load_jsonl(args.train_data)
    train_dataset = NLIDataset(train_data, tokenizer, args.max_length) # model type
    if args.distributed:
        # OK - legacy
        val_data = load_data(args.val_data, args.local_rank, args.distributed)
    else:
        val_data = load_jsonl(args.val_data)
    val_dataset = NLIDataset(val_data, tokenizer, args.max_length)
    return train_dataset, val_dataset
        
        
if __name__=='__main__':
    args  = get_args()
    seed_everything(42)
    os.makedirs(args.output_dir, exist_ok = True)
    if args.local_rank in [-1,0]:
        with open(os.path.join(args.output_dir,'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    logger1, logger2 = get_log(args)
    if args.local_rank in [-1,0]:
        logger1.info(args)
        logger2.info(args)
        
    deepspeed.init_distributed(dist_backend='nccl') ########################################################################################
    # tokenizer, model load
    ########################################################################################
    tokenizer, model = get_tokenizer_and_model(args)
    ########################################################################################
    
    ########################################################################################
    # data
    ########################################################################################
    train_dataset, val_dataset = load_datasets(args, tokenizer)
    ds_config = synchronize_config_with_deepspeed(args, train_dataset)
    dschf = HfDeepSpeedConfig(ds_config)
    optimizer_group = make_optimizer_group(model, args.decay)
    print(args)
    model, optimizer, _, _ = deepspeed.initialize(args=args, model=model, model_parameters = optimizer_group, config = ds_config) # model, optimizer, train-data, lr_scheuler
    torch.cuda.set_device(args.local_rank)
    model.cuda()
    ########################################################################################
    
    # weighted_sampling & distributed
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset) 
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,batch_size = args.batch_size, sampler = train_sampler, collate_fn = train_dataset.collate_fn)
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset,batch_size = args.batch_size, sampler = val_sampler, collate_fn = val_dataset.collate_fn)
    ########################################################################################
    
    ########################################################################################
    # train
    ########################################################################################
    train()
    ########################################################################################
