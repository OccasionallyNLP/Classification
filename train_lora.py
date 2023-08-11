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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_constant_schedule_with_warmup, AutoModelForSeq2SeqLM
import argparse
from utils.data_utils import *
from utils.distributed_utils import *
from utils.utils import *
from sklearn.metrics import accuracy_score, f1_score

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
    parser.add_argument('--n_labels', type=int)
    
    # logging 관련
    parser.add_argument('--logging_term', type=int, default = 100)
   
    # 학습 관련
    parser.add_argument('--epochs', type=int, default = 20)
    parser.add_argument('--eval_epoch', type = int, default = 1, help = 'term of evaluation')
    parser.add_argument('--batch_size', default = 8, type=int)
    parser.add_argument('--lr', type=float, default = 5e-5)
    parser.add_argument('--warmup', type=float, default = 0.05)
    parser.add_argument('--decay', type=float, default = 0.1)
    parser.add_argument('--accumulation_steps', type=int, default = 1) # 221124 추가
    
    # peft
    parser.add_argument('--r', type=int, default = 8)
    parser.add_argument('--lora_alpha', type=int, default = 16)
    parser.add_argument('--lora_dropout', type=float, default = 0.1)
    
    
    # 경량화
    parser.add_argument('--fp16', type=str2bool, default = False)
    parser.add_argument('--fp16_model', type=str2bool, default = False)
    
    # PTM model
    parser.add_argument('--ptm_path', type=str)
    parser.add_argument('--model_path', type=str)
    
    # model input
    parser.add_argument('--max_length', type=int)
    parser.add_argument('--answer_max_length', type=int, default = 20)
    # distributed 관련
    parser.add_argument('--local_rank', type=int, default = -1)
    parser.add_argument('--distributed', type=str2bool, default = False)
    parser.add_argument('--early_stop', type=str2bool, default = True) # XXX220919
    parser.add_argument('--early_stop_metric', type=str, default = 'loss') # 230619 추가
    parser.add_argument('--early_stop_metric_is_max_better', type=str2bool, default = False) # 230619 추가
    parser.add_argument('--patience', type=int, default = 3)
    parser.add_argument('--save_model_every_epoch', type=str2bool, default = False) # 230619 추가
    
    args  = parser.parse_args()
    return args

def train():
    # optimizer
    optimizer_grouped_parameters = make_optimizer_group(model, args.decay)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.decay)
    # scheduler
    t_total = len(train_dataloader)*args.epochs//args.accumulation_steps
    n_warmup = int(t_total*args.warmup) if args.warmup<1 else int(args.warmup)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=n_warmup)
    if args.local_rank in [-1,0]:
        early_stop = EarlyStopping(args.patience, args.output_dir, max = args.early_stop_metric_is_max_better, min_difference=1e-5)
    if args.fp16:
        scaler = GradScaler()
    flag_tensor = torch.zeros(1).cuda()
    ########################################################################################
    # train
    ########################################################################################
    global_step = 0
    optimizer.zero_grad()      
    for epoch in range(1, args.epochs+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.
        step = 0
        iter_bar = tqdm(train_dataloader, desc='step', disable=args.local_rank not in [-1,0])
        #train
        for data in iter_bar:
            step+=1
            data = {i:j.cuda() for i,j in data.items() if i!='token_type_ids'}
            if 't5' in args.ptm_path:
                data['labels'][data['labels']==tokenizer.pad_token_id]=-100 # 굉장히 중요.
            
            if args.fp16_model or not args.fp16:
                output = model.forward(**data)
                loss = output.loss
                loss = loss / args.accumulation_steps
                loss.backward()
                if step%args.accumulation_steps==0 or (
                    len(train_dataloader) <= args.accumulation_steps
                    and (step) == len(train_dataloader)
            ):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step+=1
            else:
                with autocast():
                    output = model.forward(**data)
                    loss = output.loss
                    loss = loss / args.accumulation_steps
                    scaler.scale(loss).backward()
                    if step%args.accumulation_steps==0 or (
                    len(train_dataloader) <= args.accumulation_steps
                    and (step) == len(train_dataloader)
            ):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step+=1
                
            if args.distributed:
                torch.distributed.reduce(loss, 0)
                loss = loss / torch.distributed.get_world_size()
            epoch_loss+=loss.item()*args.accumulation_steps
            iter_bar.set_postfix({'epoch':epoch, 'global_step':global_step, 'step':step, 'lr':f"{scheduler.get_last_lr()[0]:.5f}",'epoch_loss':f'{epoch_loss/step:.5f}'}) 
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
            val_scores_ = evaluation(args, model, tokenizer, val_dataloader)
            val_scores = get_scores(args.local_rank, val_scores_, args.distributed)            
            
            if args.local_rank in [-1,0]:
                logger1.info(f'Val ---- epoch : {epoch} ----- scores:{val_scores}')
                logger2.info(f'Val ---- epoch : {epoch} ----- scores:{val_scores}')
                model_to_save = model.module if hasattr(model,'module') else model
                if args.save_model_every_epoch:
                    cur_path = os.path.join(args.output_dir,'model_%d'%epoch)
                    os.makedirs(cur_path, exist_ok = True)
                    model_to_save.save_pretrained(cur_path)
                    torch.save(optimizer.state_dict(), os.path.join(cur_path, "optimizer_%d.pt"%epoch))
                    torch.save(scheduler.state_dict(), os.path.join(cur_path, "scheduler_%d.pt"%epoch))
                early_stop.check(model_to_save, val_scores[args.early_stop_metric])  
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
        cur_path = os.path.join(args.output_dir,'best_model')
        os.makedirs(cur_path, exist_ok = True)
        model_to_save = model.module if hasattr(model,'module') else model
        model_to_save.save_pretrained(cur_path)
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
    if 't5' in args.ptm_path:
        model =  AutoModelForSeq2SeqLM.from_pretrained(args.ptm_path, num_labels=args.n_labels, torch_dtype="auto", low_cpu_mem_usage=True, pad_token_id=tokenizer.pad_token_id)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.ptm_path, num_labels=args.n_labels, torch_dtype="auto", low_cpu_mem_usage=True, pad_token_id=tokenizer.pad_token_id)
    model.resize_token_embeddings(len(tokenizer))
    if args.model_path is not None:
        model_state_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(model_state_dict)
    
    # peft
    if 'bert' in args.ptm_path or 'roberta' in args.ptm_path:
        peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=args.r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    elif 't5' in args.ptm_path:
        peft_config = LoraConfig(task_type="SEQ2SEQ_LM", inference_mode=False, r=args.r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    else:
        peft_config = LoraConfig(task_type="CAUSAL_LM", inference_mode=False, r=args.r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
        
    model = get_peft_model(model, peft_config)
    print('model trainable parameters')
    model.print_trainable_parameters()
    return tokenizer, model 


def load_datasets(args, tokenizer):
    # LOAD DATASETS
    train_data = load_jsonl(args.train_data)
    if 't5' in args.ptm_path:
        train_dataset = T5NLIDataset(train_data, tokenizer, args.max_length) # model type
    else:
        train_dataset = NLIDataset(train_data, tokenizer, args.max_length) # model type
    if args.distributed:
        # OK - legacy
        val_data = load_data(args.val_data, args.local_rank, args.distributed)
    else:
        val_data = load_jsonl(args.val_data)
    if 't5' in args.ptm_path:
        val_dataset = T5NLIDataset(val_data, tokenizer, args.max_length)
    else:        
        val_dataset = NLIDataset(val_data, tokenizer, args.max_length)
    return train_dataset, val_dataset
        
        
if __name__=='__main__':
    args  = get_args()
    seed_everything(42)
    os.makedirs(args.output_dir, exist_ok = True)
    
    logger1, logger2 = get_log(args)
    if args.local_rank in [-1,0]:
        logger1.info(args)
        logger2.info(args)
        
    ########################################################################################
    # tokenizer, model load
    ########################################################################################
    #peft_config = get_peft_config(args)
    tokenizer, model = get_tokenizer_and_model(args)
    ########################################################################################
    
    ########################################################################################
    # distributed 관련
    ########################################################################################
    args.local_rank = int(os.environ["LOCAL_RANK"])
    if args.distributed:
        assert torch.cuda.is_available()
        assert torch.cuda.device_count()>1
        # 이 프로세스가 어느 gpu에 할당되는지 명시
        torch.cuda.set_device(args.local_rank)
        # 통신을 위한 초기화
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],output_device = args.local_rank)
    else:
        model.cuda()
    ########################################################################################
    
    ########################################################################################
    # data
    ########################################################################################
    train_dataset, val_dataset = load_datasets(args, tokenizer)
    
    # save
    if args.local_rank in [-1,0]:
        with open(os.path.join(args.output_dir,'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
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
