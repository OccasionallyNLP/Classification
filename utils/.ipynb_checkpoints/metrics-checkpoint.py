# -*- coding: utf-8 -*-
# metrics
import numpy as np
from collections import Counter
import string
import re
import argparse
import json
import sys
import os
from bs4 import BeautifulSoup
from typing import List
from rouge_metric import PyRouge
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from nltk.util import ngrams
import numpy as np

# scores
# # accuracy
# # 개수에 따른 accuracy

def hit(actual:List[int],predict:List[List[int]])->dict:
    from collections import defaultdict
    result = defaultdict(list)
    for i,j in zip(actual, predict):
        for k in range(1,101):
            result[k].append(i in j[:k])
    output = dict()
    for i,j in result.items():
        output[i]=float(np.round(sum(j)/len(j),3))
    print(output)
    return output

def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

def post_process(input_string:str):
    SPAN_TOKEN = '<extra_id_%s>'
    UNUSED_TOKEN = 'UNUSED'
    for i in range(100):
        input_string = re.sub(SPAN_TOKEN%i, '', input_string).strip()
    for i in range(5000):
        input_string = re.sub(UNUSED_TOKEN+str(i).zfill(4), '', input_string).strip()
    return input_string

def normalize_answer(s):    
    def tag_clean(t):
        return BeautifulSoup(t).get_text()

    def remove_(text):
        ''' 불필요한 기호 제거 '''
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub('《', " ", text)
        text = re.sub('》', " ", text)
        text = re.sub('<', " ", text)
        text = re.sub('>', " ", text) 
        text = re.sub('〈', " ", text)
        text = re.sub('〉', " ", text)   
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)      
        return text

    def white_space_fix(text):
        return ' '.join(text.split()).replace('\n','').replace('\t','').replace(' ','')

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(tag_clean(s)))))

# char f1
def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
   
    #F1 by character
    prediction_Char = []
    for tok in prediction_tokens:
        now = [a for a in tok]
        prediction_Char.extend(now)
        
    ground_truth_Char = []
    for tok in ground_truth_tokens:
        now = [a for a in tok]
        ground_truth_Char.extend(now)   
        
    common = Counter(prediction_Char) & Counter(ground_truth_Char)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_Char)
    recall = 1.0 * num_same / len(ground_truth_Char)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

# token unigram f1 score
def unigram_f1_score(prediction, ground_truth, tokenizer):
    if tokenizer is None:
        prediction_tokens = prediction.split()
        gt_tokens = ground_truth.split()
        
    else:
        prediction_tokens = tokenizer(prediction)
        gt_tokens = tokenizer(ground_truth)

    common = Counter(prediction_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def sentence_rouge_l(actual, predict, tokenizer=None):
    rouge = PyRouge(rouge_n = (1,2,4), rouge_l = True, rouge_w = True, rouge_w_weight = 1.2, rouge_s = True, rouge_su = True, skip_gap = 4)
    if tokenizer is not None:
        hypothesis = [[tokenizer(predict)]]
        reference = [[[tokenizer(actual)]]]
    else:
        hypothesis = [[predict.split()]]
        reference = [[[actual.split()]]]
    score = rouge.evaluate_tokenized(hypothesis, reference)
    return score['rouge-l']['f']

def sentence_bleu_score(actual, predict, tokenizer=None):
    if tokenizer is not None:
        hypothesis = tokenizer(predict)
        reference = tokenizer(actual)
    else:
        hypothesis = predict.split()
        reference = actual.split()
    cc = SmoothingFunction()
    score1 = sentence_bleu([reference],hypothesis,weights=(1,0,0,0))#, smoothing_function = cc.method4)
    score4 = sentence_bleu([reference],hypothesis,weights=(0,0,0,1), smoothing_function = cc.method4)
    return score1, score4

# exact match score
def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

# retriever 관련
def compute_topk_accuracy(answers:List[int], candidates:List[List[int]])->List[float]:
    # answers : 정답지 - 문서의 id로 구성됨.
    # candidates : 2 dimension List
    # shape : (N, k) <- 즉 k개의 정답지에 대한 id를 지니고 있음.
    answers = np.array(answers).reshape(-1,1)
    candidates = np.array(candidates)
    N,k = candidates.shape
    
    acc_score = []
    for j in range(1,k+1):
        a = np.sum(np.sum(candidates[:,:j] == answers, axis=-1)>=1)/N
        acc_score.append(a)
    return acc_score

## precision <- 어차피 prediction(precision에서 분모 값)
def compute_topk_precision(answers:List[int], candidates:List[List[int]])->float:
    # answers : 정답지 - 문서의 id로 구성됨.
    # candidates : 2 dimension List
    # shape : (N, k) <- 즉 k개의 정답지에 대한 id를 지니고 있음.
    answers = np.array(answers).reshape(-1,1)
    candidates = np.array(candidates)
    N,k = candidates.shape
    precision_score = []
    for j in range(1,k+1):
        a = (np.sum(candidates[:,:j] == answers, axis=-1))/j
        precision_score.append(np.sum(a)/N)
    return precision_score

## MRR - Mean Reciprocal Rank
## 1/|Q|*sum(1/RANK_i)
def compute_MRR_K(answers:List[int], candidates:List[List[int]],K:int)->float:
    # answers : 정답지 - 문서의 id로 구성됨.
    # candidates : 2 dimension List
    # shape : (N, k) <- 즉 k개의 정답지에 대한 id를 지니고 있음.
    answers = np.array(answers).reshape(-1,1)
    candidates = np.array(candidates)
    candidates = candidates[:,:K]
    N,k = candidates.shape
    assert K<=k
        
    a = (candidates == answers)
    hit = a.sum(axis=-1)>=1
    rank = (np.argmax(a,axis=-1)+1).astype(np.float)
    #print(rank)
    reciprocal = np.reciprocal(rank)*hit
    #print(reciprocal)
    #print(hit)    
    return np.sum(reciprocal)/N

# dist_n
def distinct_n_sentence_level(sentence, n, tokenizer=None):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if tokenizer is None:
        tokens = sentence.split()
    else:
        tokens = tokenizer.tokenize(sentence)
    if len(tokens) == 0:
        return 0.0  # Prevent a zero division
    n_grams = list(ngrams(tokens, n))
    distinct_ngrams = set(n_grams)
    if len(n_grams)==0:
        return 0.0
    return len(distinct_ngrams) / len(n_grams)


# ndcg
def ndcg(rel_true, rel_pred, p=None):
    """ Returns normalized Discounted Cumulative Gain
    Args:
        rel_true (1-D Array): relevance lists for particular user, (n_songs,)
        rel_pred (1-D Array): predicted relevance lists, (n_pred,)
        p (int): particular rank position
        form (string): two types of nDCG formula, 'linear' or 'exponential'
    Returns:
        ndcg (float): normalized discounted cumulative gain score [0, 1]
        ndcg - 1에 가까울수록 good
    """
    rel_true = np.sort(rel_true)[::-1]
    if p is None:
        p = min([len(rel_true), len(rel_pred)])
    discount = 1 / (np.log2(np.arange(p) + 1 + 1))

    idcg = np.sum(rel_true[:p] * discount)
    dcg = np.sum(rel_pred[:p] * discount)
    return dcg / idcg


# if __name__ == "__main__":
#     song_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8}
#     user_lists = ["USER1", "USER2", "USER3"]

#     relevance_true = {
#         "USER1": [3, 3, 2, 2, 1, 1, 0, 0, 0],
#         "USER2": [3, 2, 1, 1, 2, 0, 1, 1, 1],
#         "USER3": [0, 1, 0, 1, 2, 3, 3, 1, 0]
#     }

#     s1_prediction = {
#         "USER1": ['A', 'E', 'C', 'D', 'F'],
#         "USER2": ['G', 'E', 'A', 'B', 'D'],
#         "USER3": ['C', 'G', 'F', 'B', 'E']
#     }

#     s2_prediction = {
#         "USER1": ['A', 'B', 'C', 'G', 'E'],
#         "USER2": ['B', 'A', 'G', 'E', 'F'],
#         "USER3": ['E', 'G', 'F', 'B', 'I']      
#     }

#     for user in user_lists:
#         print(f'===={user}===')
#         r_true = relevance_true[user]

#         s1_pred = [r_true[song_index[song]] for song in s1_prediction[user]]
#         s2_pred = [r_true[song_index[song]] for song in s2_prediction[user]]
   
#         print(f'S1 nDCG@5 (linear): {ndcg(r_true, s1_pred, 5)}')
#         print(f'S2 nDCG@5 (linear): {ndcg(r_true, s2_pred, 5)}')
        