#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2022-05-11 11:12:17.938441
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 

class Dataset(mt.Dataset):
  def __init__(self, subset='valid', **kwargs):
    super(Dataset, self).__init__(subset, **kwargs)

  def parse(self, example):
    keys = []
    excl_keys = [] 
    dynamic_keys = ['input_ids', 'attention_mask', 'token_type_ids'] 
    self.auto_parse(keys=keys, exclude_keys=excl_keys + dynamic_keys)
    self.adds(dynamic_keys)
    fe = self.parse_(serialized=example)
          
    mt.try_append_dim(fe)
    
    x = fe
    y = fe['label']
     
    return x, y  
