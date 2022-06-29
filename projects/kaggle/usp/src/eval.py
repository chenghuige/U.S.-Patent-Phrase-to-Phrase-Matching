#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   eval.py
#        \author   chenghuige  
#          \date   2022-05-11 11:12:10.236762
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
from src.postprocess import *

def calc_metric(y_true, y_pred):
  return scipy.stats.pearsonr(y_true, y_pred)[0]

def evaluate(y_true, y_pred, x, other, is_last=False):
  res = {}
  
  eval_dict = gezi.get('eval_dict')
  if eval_dict:
    x.update(eval_dict)
    
  x.update(other)
  # if is_last:
  gezi.set('eval:x', x)
    
  y_pred = to_pred(y_pred)
  ic(np.asarray(list(zip(y_true, y_pred))), y_true.mean(), y_pred.mean())
  score = calc_metric(y_true, y_pred)
  res['score'] = score

  return res

def valid_write(x, label, predicts, ofile, others={}):
  ofile = f'{FLAGS.model_dir}/valid.csv'
  write_result(x, predicts, ofile, others, is_infer=False)

def infer_write(x, predicts, ofile, others={}):
  ofile = f'{FLAGS.model_dir}/submission.csv'
  write_result(x, predicts, ofile, others, is_infer=True)
  
def to_df(x):
  predicts = to_pred(x['pred'])
  
  m = {
    'id': x['id'],
    'score': predicts
  }
  df = pd.DataFrame(m)
  return df
      
def write_result(x, predicts, ofile, others={}, is_infer=False):
  if is_infer:
    m = gezi.get('infer_dict')
  else:
    m = gezi.get('eval_dict')
    
  if m:
    x.update(m)
  x.update(others)

  df = to_df(x)
  if not is_infer:
    df['label'] = x['label']
  ic(df)
  df.to_csv(ofile, index=False)
  