#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige  
#          \date   2022-05-11 11:12:57.220407
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 

from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig

from src.config import *
from src.preprocess import *

class Model(nn.Module):
  
  def __init__(self, **kwargs):
    super().__init__(**kwargs)  
            
    self.backbone, self.tokenizer = self.init_backbone(FLAGS.backbone)
    
    config = self.backbone.config
    dim = config.hidden_size
        
    if FLAGS.rnn:
      RNN = getattr(nn, FLAGS.rnn_type)
      if not FLAGS.rnn_bi:
        self.seq_encoder = RNN(dim, dim, FLAGS.rnn_layers, dropout=FLAGS.rnn_dropout, bidirectional=False, batch_first=True)
      else:
        if not FLAGS.rnn_double_dim:
          self.seq_encoder = RNN(dim, int(dim / 2), FLAGS.rnn_layers, dropout=FLAGS.rnn_dropout, bidirectional=True, batch_first=True)
        else:
          self.seq_encoder = RNN(dim, dim, FLAGS.rnn_layers, dropout=FLAGS.rnn_dropout, bidirectional=True, batch_first=True)
          dim *= 2
          
    self.pooling = lele.layers.Pooling(FLAGS.pooling, dim)
    dim = self.pooling.output_dim
    if FLAGS.concat_last:
      dim = config.hidden_size * 4
    
    Linear = nn.Linear
    
    odim = 1
    if FLAGS.method == 'cls':
      odim = 5
    self.dense = Linear(dim, odim)
    
    if FLAGS.mean_score_rate:
      self.pooling2 = lele.layers.Pooling(FLAGS.pooling, dim)
      self.dense2 = Linear(dim, 1)
    
    if FLAGS.layer_norm:
      self.layer_norm = torch.nn.LayerNorm(dim, eps=1e-6) 
      
    if FLAGS.drop_rate > 0:
      self.dropout = nn.Dropout(FLAGS.drop_rate)
    
    if FLAGS.opt_fused:
      lele.replace_with_fused_layernorm(self)
  
  def init_backbone(self, backbone_name, model_dir=None, load_weights=False):
    model_dir = model_dir or FLAGS.model_dir
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    if FLAGS.prompt:
      # might need more investigation for prompt seems not get better result
      from transformers import AutoModelForPreTraining as AutoModel
    if 'cocolm' in backbone_name:
      if not FLAGS.prompt:
        from cocolm.modeling_cocolm import COCOLMModel as AutoModel
      else:
        from cocolm.modeling_cocolm import COCOLMPreTrainedModel as AutoModel
      
      from cocolm.configuration_cocolm import COCOLMConfig as AutoConfig

    try:
      config = AutoConfig.from_pretrained(model_dir)
    except Exception:
      try:
        config = AutoConfig.from_pretrained(backbone_name)
      except Exception:
        backbone_ = (backbone_name.split('/')[-1]).replace('_', '-')
        config = AutoConfig.from_pretrained(f'../input/{backbone_}')
    if FLAGS.concat_last:
      config.update({'output_hidden_states':True})
      
    self.config = config
    model_dir = model_dir or FLAGS.model_dir
    ic(model_dir, os.path.exists(f'{model_dir}/model.pt'))
    
    if os.path.exists(f'{model_dir}/model.pt'):
      backbone = AutoModel.from_config(config)
    else:
      backbone = AutoModel.from_pretrained(backbone_name, config=config)
      
    if os.path.exists(f'{model_dir}/config.json'):
      tokenizer = AutoTokenizer.from_pretrained(model_dir)
    else:
      tokenizer = get_tokenizer(backbone_name)
    try:
      backbone.resize_token_embeddings(len(tokenizer)) 
    except Exception:
      pass
    
    if FLAGS.gradient_checkpointing:
      backbone.gradient_checkpointing_enable()
    
    if FLAGS.freeze_emb:
      ## not much difference, but used for submission
      # if hasattr(backbone, 'embeddings'):
      #   backbone.embeddings.requires_grad_(False)
      # elif hasattr(backbone, 'shared'):
      #   backbone.shared.requires_grad_(False)
      lele.freeze(lele.get_word_embeddings(backbone))
    
    # lele.freeze(backbone)
    return backbone, tokenizer 
  
  def encode(self, inputs):
    backbone, tokenizer = self.backbone, self.tokenizer

    m = {
      'input_ids': inputs['input_ids'],
      'attention_mask': inputs['attention_mask'],
    } 
    if 'token_type_ids' in inputs:
      m['token_type_ids'] = inputs['token_type_ids']
      
    x = backbone(**m)
    key = 'last_hidden_state' if not FLAGS.concat_last else 'hidden_states'
    if key in x:
      x = x[key]
    else:
      x = x[0]
    
    if FLAGS.rnn:
      x, _ = self.seq_encoder(x)

    return x
  
  def pool(self, x, inputs):
    if not FLAGS.concat_last:
      mask = inputs['attention_mask'] if FLAGS.pooling_mask else None
      x = self.pooling(x, mask)
    else:
      x = torch.cat([x[-1][:,0], x[-2][:,0], x[-3][:,0], x[-4][:,0]], -1)
    return x
  
  def fake_pred(self, inputs, requires_grad=False):
    input_ids =  inputs['input_ids'] if not 0 in inputs else inputs[0]['input_ids']
    bs = input_ids.shape[0] 
    return torch.rand([bs, 1], device=input_ids.device, requires_grad=requires_grad)
    
  def forward(self, inputs):
    if FLAGS.fake_infer:
      return {
        'pred': self.fake_pred(inputs, requires_grad=self.training),
      }
    
    res = {}
    
    if not FLAGS.two_tower:
      xs = self.encode(inputs)
      if not FLAGS.prompt:
        x = self.pool(xs, inputs)
        if FLAGS.layer_norm:
          x = self.layer_norm(x)
        res['pred'] = self.dense(x) 
      else:
        x = xs[:, inputs['prompt_idx'][0]]
        yes_id = self.tokenizer.convert_tokens_to_ids('Yes')
        res['pred'] = x[:, yes_id]
        
      ## bad result
      if FLAGS.mean_score_rate:
        x2 = self.pooling2(xs, inputs['attention_mask'])
        res['pred2'] = self.dense2(x2)
      
      ## not used
      if 'input_ids2' in inputs:
        inputs['input_ids'] = inputs['input_ids2']
        inputs['attention_mask'] = inputs['attention_mask2']
        if 'token_type_ids2' in inputs:
          inputs['token_type_ids'] = inputs['token_type_ids2']
        x = self.encode(inputs)
        x = self.pool(x, inputs)
        # dropout not used, if used here might need training=? maybe still prefer to use 
        if FLAGS.drop_rate > 0:
          x = self.dropout(x)
        res['pred2'] = self.dense(x)
    else:
      ## bad result for usint two_tower
      m = {'input_ids': inputs['anchor_ids'], 'attention_mask': inputs['anchor_mask']}
      x1 = self.encode(m)
      x1 = self.pooling(x1, m['attention_mask'])
      if FLAGS.layer_norm:
        x1 = self.layer_norm(x1)
      m = {'input_ids': inputs['target_ids'], 'attention_mask': inputs['target_mask']}
      x2 = self.encode(m)
      x2 = self.pooling(x2, m['attention_mask'])
      if FLAGS.layer_norm:
        x2 = self.layer_norm(x2)
      res['pred'] = torch.einsum('ij,ij->i', x1, x2)
      # res['pred'] = torch.cosine_similarity(x1, x2)
      
    if not 'pred' in res:
      res['pred'] = self.fake_pred(inputs)
    
    if FLAGS.sigmoid:
      res['pred'] = torch.sigmoid(res['pred'])
    
    return res

  def get_loss_fn(self):
    from src.torch.loss import calc_loss
    return calc_loss
