#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2022-05-11 11:13:01.390146
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
from torch import nn

def calc_loss(res, y, x, step=None, epoch=None, training=None):
  scalars = {}
        
  y = x['label'] 
  if FLAGS.method == 'cls':
    y = x['cat']  
  if FLAGS.transform_score:
    y = x['relevance']
  y_ = res[FLAGS.pred_key]
  reduction = 'mean'
  loss = 0
  odim = 1

  if FLAGS.method == 'soft_binary':
    loss_obj = nn.BCEWithLogitsLoss(reduction=reduction)
    y = y.float() #BCE need float label
    base_loss = loss_obj(y_.view(-1), y.view(-1))
  elif FLAGS.method == 'cls':
    loss_obj = nn.CrossEntropyLoss(reduction=reduction)  
    base_loss = loss_obj(y_.view(-1, 5), y.view(-1))
  elif FLAGS.method == 'mse':
    loss_obj = nn.MSELoss(reduction=reduction)
    y = y.float()
    base_loss = loss_obj(y_.view(-1), y.view(-1))
  elif FLAGS.method == 'mse2':
    loss_obj = nn.MSELoss(reduction=reduction)
    y = y.float()
    y_ = nn.functional.sigmoid(y_)
    base_loss = loss_obj(y_.view(-1), y.view(-1))
  elif FLAGS.method == 'pearson':
    y = y.float()
    # if 'pred2' in res:
    #   y2_ = res['pred2']
    #   y2 = x['label2'].float()
    #   y_ = torch.cat([y_, y2_], 0)
    #   y = torch.cat([y, y2], 0)
    base_loss = 1. - lele.losses.pearson(y_.view(-1), y.view(-1))
      
  elif FLAGS.method == 'pearson2':
    y = y.float()
    y_ = nn.functional.sigmoid(y_)
    base_loss = 1. - lele.losses.pearson(y_.view(-1), y.view(-1))
  
  scalars['loss/base'] = base_loss.detach().cpu().numpy()
  loss += base_loss
  
  if FLAGS.mse_loss_rate > 0:
    loss_obj = nn.MSELoss(reduction=reduction)
    y = y.float()
    mse_loss = loss_obj(y_.view(-1), y.view(-1))
    mse_loss *= FLAGS.mse_loss_rate
    scalars['loss/mse'] = mse_loss.detach().cpu().numpy()
    loss += mse_loss
  
  if FLAGS.binary_loss_rate > 0:
    loss_obj = nn.BCEWithLogitsLoss(reduction=reduction)
    y = y.float() #BCE need float label
    binary_loss = loss_obj(y_.view(-1), y.view(-1))
    binary_loss *= FLAGS.binary_loss_rate
    scalars['loss/binary'] = binary_loss.detach().cpu().numpy()
    loss += binary_loss
  
  if FLAGS.mean_score_rate > 0:
    loss_obj = nn.MSELoss(reduction=reduction)
    mean_score_loss = loss_obj(res['pred2'].float().view(-1), x['mean_score'].float().view(-1))
    mean_score_loss *= FLAGS.mean_score_rate
    scalars['loss/mean_score'] = mean_score_loss.detach().cpu().numpy()
    loss += mean_score_loss
  
  loss *= FLAGS.loss_scale
  
  if FLAGS.rdrop_rate > 0:
    ## FIXME TODO for electra not work so [electra/roberta] why? not just use awp no rdrop, deberta/bart all ok
    # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.LongTensor [1, 229]] is at version 3;
    # expected version 2 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
    def rdrop_loss(p, q):
      rloss = 0.
      # rloss += lele.losses.compute_kl_loss(p['pred'], q['pred'])
      rloss += lele.losses.compute_kl_loss(p['pred'].view(1, -1), q['pred'].view(1, -1))
      return rloss
    gezi.set('rdrop_loss_fn', lambda p, q: rdrop_loss(p, q))
          
  lele.update_scalars(scalars, training)
  
  return loss
  
