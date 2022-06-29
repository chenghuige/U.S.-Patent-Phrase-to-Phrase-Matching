#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   postprocess.py
#        \author   chenghuige  
#          \date   2022-05-11 11:17:37.027553
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib import scale

from gezi.common import * 
from src.config import *
import sklearn

def normalize(pred):
  # from sklearn.preprocessing import MinMaxScaler
  pred = np.asarray(pred)
  #ic(pred, pred.min(), pred.max(), pred.mean())
  # pred = np.clip(pred, -1, 2)
  scaler = sklearn.preprocessing.MinMaxScaler()
  # scaler = sklearn.preprocessing.StandardScaler()
  # scaler = sklearn.preprocessing.RobustScaler()
  # scaler = sklearn.preprocessing.PowerTransformer()
  scores = scaler.fit_transform(pred.reshape(-1, 1)).reshape(-1)
  #ic(scores, scores.min(), scores.max(), scores.mean())
  return scores

def transform(score, thre=0.05):
  thres = [0, 0.25, 0.5, 0.75, 1]
  for thre_ in thres:
    if abs(score - thre_) < thre:
      return thre_
  return score
  
def to_pred(logits):
  logits = np.asarray(logits)
  if FLAGS.method == 'soft_binary':
    probs = gezi.sigmoid(logits)
    scores = probs 
    if FLAGS.cluster_score:
      scores = (probs * 4. + 0.5).astype(int)
  elif FLAGS.method == 'cls':
    probs = gezi.softmax(logits)
    scores = np.argmax(probs, -1)
  elif FLAGS.method == 'mse':
    # scores = (logits * 4. + 0.5).astype(int)
    scores = logits
  elif FLAGS.method == 'mse2':
    probs = gezi.sigmoid(logits)
    scores = probs
  elif FLAGS.method == 'pearson':
    # probs = gezi.sigmoid(logits) 
    # scores = probs
    scores = logits
  elif FLAGS.method == 'pearson2':
    probs = gezi.sigmoid(logits) 
    scores = probs
  else:
    scores = logits
  # ic(scores, scores.max(), scores.min(), scores.mean())
  scores = normalize(scores)
  # ic(scores, scores.max(), scores.min(), scores.mean())
  # scores = np.asarray([transform(x) for x in scores])
  return scores
