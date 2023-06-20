import torch
import torch.nn as nn
import torch.nn.functional as F
def compute_kl_loss(p:torch.tensor, q:torch.tensor):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
    loss = (p_loss + q_loss) / 2
    return loss
