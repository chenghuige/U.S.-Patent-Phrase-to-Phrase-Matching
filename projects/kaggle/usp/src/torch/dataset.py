#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2022-05-31 03:21:26.236767
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
from torch.utils.data import Dataset as TorchDataset
from src.config import *
from src.preprocess import *

class Dataset(TorchDataset):
  def __init__(self, subset='valid'):
    self.subset = subset
    self.mark = 'train' if subset in ['train', 'valid'] else 'test'
    df = get_df(self.mark)
    self.df = df
    if FLAGS.fold_seed2 is None:
      if subset == 'valid':
        self.df = df[df.fold==FLAGS.fold]
      if not FLAGS.online:
        if subset == 'train':
          self.df = df[df.fold!=FLAGS.fold]
    else:
      if subset == 'valid':
        self.df = df[df.fold==-1]
      if not FLAGS.online:
        if subset == 'train':
          self.df = df[(df.fold != FLAGS.fold) & (df.fold != -1)]
      else:
        if subset == 'train':
          self.df = df[df.fold != -1]
    
    ic(df, FLAGS.aug_seed)
    self.rng = np.random.default_rng(FLAGS.aug_seed)
    self.rng2 = np.random.default_rng(FLAGS.aug_seed + 1)

  def __getitem__(self, idx):
    row = dict(self.df.iloc[idx])
    
    if FLAGS.shuffle_targets and (self.subset == 'train' or FLAGS.shuffle_valid_targets):
      targets_idx = FLAGS.targets_idx
      text = row['text']
      texts = text.split('[SEP]')
      
      targets = texts[targets_idx]
      targets = targets.split('; ')
      self.rng.shuffle(targets)
      targets = '; '.join(targets)
      texts[targets_idx] = targets
      
      if FLAGS.use_sector:
        targets_idx += 1
        targets = texts[targets_idx]
        targets = targets.split('; ')
        self.rng.shuffle(targets)
        targets = '; '.join(targets)
        texts[targets_idx] = targets
      
      if FLAGS.use_anchor:
        targets_idx += 1
        targets = texts[targets_idx]
        targets = targets.split('; ')
        self.rng.shuffle(targets)
        targets = '; '.join(targets)
        texts[targets_idx] = targets  
      
      row['text'] = '[SEP]'.join(texts)
    
    if self.subset == 'train':
      if FLAGS.shuffle_context_prob > 0 and self.rng.random() < FLAGS.shuffle_context_prob:
        text = row['text']
        texts = text.split('[SEP]')
        context = texts[-2]
        context = context.split('; ')
        self.rng.shuffle(context)
        context = '; '.join(context)
        texts[-2] = context
        row['text'] = '[SEP]'.join(texts)
        
      if FLAGS.aug_prob > 0 and self.rng.random() < FLAGS.aug_prob:
        text = row['text']
        texts = text.split('[SEP]')
        targets = texts[-1]
        targets = targets.split('; ')
        num_targets = len(targets)
        if num_targets:
          mask = self.rng.random(size=num_targets) < FLAGS.mask_prob
          # if FLAGS.mask_self:
          #   targets_ = [targets[i] for i in range(num_targets) if not mask[i]]
          # else:
          #   targets_ = [targets[i] for i in range(num_targets) if (not mask[i]) or (targets[i] == row['target'])]
          # ic(len(targets), len(targets_))
          # targets = '; '.join(targets_)
          for i in range(num_targets):
            if mask[i]:
              targets[i] = '[MASK]'
          targets = '; '.join(targets)
          texts[-1] = targets
          row['text'] = '[SEP]'.join(texts)
    
    fe = parse_example(row)
    
    if self.subset == 'train' and FLAGS.sample_targets:
      if row['targets_list']:
        text = row['text']
        texts = text.split('[SEP]')
        idx = self.rng2.integers(len(row['targets_list']))
        texts[1] = row['targets_list'][idx]
        row['text'] = '[SEP]'.join(texts)
      
      fe2 = parse_example(row)  
      fe['input_ids2'] = fe2['input_ids']
      fe['attention_mask2'] = fe2['attention_mask']
      if 'token_type_ids' in fe2:
        fe['token_type_ids2'] = fe2['token_type_ids']
      anchor = row['anchor']
      target = texts[1]
      fe['label2'] = pair_scores[f'{anchor}\t{target}']
        
    return fe, fe['label']
    
  def __len__(self):
    return len(self.df)
  