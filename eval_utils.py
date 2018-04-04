from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function 

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
from collections import OrderedDict
from cocoeval import COCOScorer
def build_sample_pairs(samples, vidIDs):
    D = OrderedDict()
    for sample, vidID in zip(samples, vidIDs):
        D[vidID] = [{'image_id': vidID, 'caption': sample}]
    return D
    
def score_with_cocoeval(samples_test, engine, ids):
    scorer = COCOScorer()
    gts_test = OrderedDict()
    for vidID in ids:
        gts_test[vidID] = engine.CAP[vidID]
    test_score = scorer.score(gts_test, samples_test, ids)
    return test_score
    

def eval_split(model, crit, loader, eval_kwargs={}):
    
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                if w in loader.seg_ix:
                    continue
                ww.append(loader.ix_to_word[1]
                          if w > len(loader.ix_to_word) else loader.ix_to_word[w])
            capsw.append(' '.join(ww))
        return capsw

    
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 100)
    remove_len = eval_kwargs.get('remove_len', 1)
    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    count = 0
    '''        '''
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size
        if data.get('labels', None) is not None:
            # forward the model to get loss
            tmp = [data['fc_feats'], data['labels'], data['x_phrase_mask_0'], data['x_phrase_mask_1'], \
                   data['label_masks'], data['salicy_seg'], data['seg_mask']]
            tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
            fc_feats, seq, phrase_mask_0, phrase_mask_1, masks, salicy_seg, seg_mask = tmp
            output, _ = model(fc_feats, seq, phrase_mask_0, phrase_mask_1, masks, seg_mask, remove_len)
            loss = crit(output, seq[remove_len:,:].permute(1,0), masks[remove_len:,:].permute(1,0)).data[0]
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1
        count = count + 1
        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        if verbose:
            print('evaluating validation preformance... (%f)' %( loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break
    print('evaluating sum validation preformance... (%f)' %( loss_sum/(1.0*count)))

    splits = ['test','val']
    score = {}
    score_list = []
    for split in splits:
        data_val, ids = loader.get_val_batch(split)
        samples = []
        for i in range(len(ids)):
            tmp = data_val[i]
            fc_feats = Variable(torch.from_numpy(tmp), volatile=True).cuda()
            # forward the model to also get generated samples for each image
            seq, _ = model.sample_beam(fc_feats.unsqueeze(0), eval_kwargs) 
            
            #set_trace()
            samples.append(seq[0])
        samples = _seqs2words(samples)
        with open('./save/'+split+'_samples.txt', 'w') as f: 
            print >>f, '\n'.join(samples) 
        sample_pairs = build_sample_pairs(samples, ids)
        score_split = score_with_cocoeval(sample_pairs, loader,ids)
        socre_items = score_split.keys()
        socre_items.sort()
        for score_i in socre_items:
            score_list.append(score_split[score_i])
        score[split] = score_split
    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, samples, score, score_list
