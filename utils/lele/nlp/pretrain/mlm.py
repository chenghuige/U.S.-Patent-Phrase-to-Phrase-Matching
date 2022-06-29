#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   mlm.py
#        \author   chenghuige  
#          \date   2022-04-28 09:05:19.995996
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gezi
from gezi.common import * 

"""
Modified from HuggingFace/transformers (https://github.com/huggingface/transformers/blob/0a3d0e02c5af20bfe9091038c4fd11fb79175546/src/transformers/data/data_collator.py#L102). 
It is a little bit faster cuz 
- intead of a[b] a on gpu b on cpu, tensors here are all in the same device
- don't iterate the tensor when create special tokens mask
And
- doesn't require huggingface tokenizer
- cost you only 550 µs for a (128,128) tensor on gpu, so dynamic masking is cheap   
"""
def mask_tokens(inputs, mask_token_index, vocab_size, special_token_indices, mlm_probability=0.15, replace_prob=0.1, orginal_prob=0.1, ignore_index=-100):
  """ 
  Prepare masked tokens inputs/labels for masked language modeling: (1-replace_prob-orginal_prob)% MASK, replace_prob% random, orginal_prob% original within mlm_probability% of tokens in the sentence. 
  * ignore_index in nn.CrossEntropy is default to -100, so you don't need to specify ignore_index in loss
  """
  
  device = inputs.device
  labels = inputs.clone()
  
  # Get positions to apply mlm (mask/replace/not changed). (mlm_probability)
  probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
  special_tokens_mask = torch.full(inputs.shape, False, dtype=torch.bool, device=device)
  for sp_id in special_token_indices:
    special_tokens_mask = special_tokens_mask | (inputs==sp_id)
  probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
  mlm_mask = torch.bernoulli(probability_matrix).bool()
  labels[~mlm_mask] = ignore_index  # We only compute loss on mlm applied tokens

  # mask  (mlm_probability * (1-replace_prob-orginal_prob))
  mask_prob = 1 - replace_prob - orginal_prob
  mask_token_mask = torch.bernoulli(torch.full(labels.shape, mask_prob, device=device)).bool() & mlm_mask
  inputs[mask_token_mask] = mask_token_index

  # replace with a random token (mlm_probability * replace_prob)
  if int(replace_prob)!=0:
    rep_prob = replace_prob/(replace_prob + orginal_prob)
    replace_token_mask = torch.bernoulli(torch.full(labels.shape, rep_prob, device=device)).bool() & mlm_mask & ~mask_token_mask
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=device)
    inputs[replace_token_mask] = random_words[replace_token_mask]

  # do nothing (mlm_probability * orginal_prob)
  pass

  return inputs, labels, mlm_mask

