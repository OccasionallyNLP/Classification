import torch
from tqdm import tqdm
import numpy as np
from utils.metrics import *
from utils.distributed_utils import *

# evaluation
def evaluation(args, model, tokenizer, eval_dataloader):
    total_loss = 0.
    model.eval()
    Predict = []
    Actual = []
    with torch.no_grad():
        for data in tqdm(eval_dataloader, desc = 'evaluate', disable =  args.local_rank not in [-1,0]):
            data = {i:j.cuda() for i,j in data.items()}
            output = model.forward(**data)
            if output.get('loss') is not None:
                loss = output['loss'].item()
                total_loss+=loss
            
            # TODO
            if args.eval_rank_type == 'point':
                # output shape - bs, n_rank
                predict = output['score'].argmax(dim=-1).cpu().tolist()
                actual = data['labels'].cpu().tolist()
            
            elif args.eval_rank_type == 'list':
                predict = output['score'].argsort(dim=-1).cpu().tolist()
                actual = data['labels'].argsort(dim=-1, descending=True).cpu().tolist()
            
            elif args.eval_rank_type == 'regression':
                predict = output['score'].cpu().tolist()
                actual = data['labels'].cpu().tolist()
                
            Predict.extend(predict)
            Actual.extend(actual)
    acc = []
    for i,j in zip(Predict, Actual):
        if args.eval_rank_type == 'list':
            acc.append(ndcg(j,i))
        else:
            acc.append(i==j)
    cnt = len(Predict)
    return dict(Loss=total_loss/len(eval_dataloader), cnt=cnt, acc=acc), Predict

def get_scores(local_rank, scores, distributed:bool):
    if distributed:
        cnt = sum([j.item() for j in get_global(local_rank, torch.tensor([scores['cnt']]).cuda())])
        acc = sum([sum(j) for j in get_global(local_rank, torch.tensor([scores['acc']]).cuda())])/cnt
        total_loss = [j.item() for j in get_global(local_rank, torch.tensor([scores['Loss']]).cuda())]
        total_loss = sum(total_loss)/len(total_loss) 
    else:
        acc = sum(scores['acc'])
        acc = acc/scores['cnt']
        total_loss = scores['Loss']
    # acc = acc.cpu().item()
    return dict(Loss=np.round(total_loss,3), acc=np.round(acc,3))
