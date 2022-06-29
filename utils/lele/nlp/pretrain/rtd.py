#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   rtd.py
#        \author   chenghuige  
#          \date   2022-04-28 09:08:02.807909
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gezi
from gezi.common import * 

from torch import nn
from torch.nn import functional as F
from fastai.losses import *

class RTDModel(nn.Module):
  
  def __init__(self, generator, discriminator, tokenizer, sampling='fp32_gumbel'):
    super().__init__()
    self.generator, self.discriminator = generator,discriminator
    self.gumbel_dist = torch.distributions.gumbel.Gumbel(0.,1.)
    self.tokenizer = tokenizer
    self.sampling = sampling

  def to(self, *args, **kwargs):
    "Also set dtype and device of contained gumbel distribution if needed"
    super().to(*args, **kwargs)
    a_tensor = next(self.parameters())
    device, dtype = a_tensor.device, a_tensor.dtype
    if self.sampling == 'fp32_gumbel': dtype = torch.float32
    self.gumbel_dist = torch.distributions.gumbel.Gumbel(torch.tensor(0., device=device, dtype=dtype), torch.tensor(1., device=device, dtype=dtype))

  def forward(self, masked_inputs, sentA_lenths, is_mlm_applied, labels):
    """
    masked_inputs (Tensor[int]): (B, L)
    sentA_lenths (Tensor[int]): (B, L)
    is_mlm_applied (Tensor[boolean]): (B, L), True for positions chosen by mlm probability 
    labels (Tensor[int]): (B, L), -100 for positions where are not mlm applied
    """
    attention_mask, token_type_ids = self._get_pad_mask_and_token_type(masked_inputs, sentA_lenths)
    # if c.my_model:
    #   gen_logits = self.generator(masked_inputs, attention_mask, token_type_ids, is_mlm_applied)[0]
    #   # already reduced before the mlm output layer, save more space and speed
    #   mlm_gen_logits = gen_logits # ( #mlm_positions, vocab_size)
    # else:
    gen_logits = self.generator(masked_inputs, attention_mask, token_type_ids)[0] # (B, L, vocab size)
    # reduce size to save space and speed
    mlm_gen_logits = gen_logits[is_mlm_applied, :] # ( #mlm_positions, vocab_size)
    
    with torch.no_grad():
      # sampling
      pred_toks = self.sample(mlm_gen_logits) # ( #mlm_positions, )
      # produce inputs for discriminator
      generated = masked_inputs.clone() # (B,L)
      generated[is_mlm_applied] = pred_toks # (B,L)
      # produce labels for discriminator
      is_replaced = is_mlm_applied.clone() # (B,L)
      is_replaced[is_mlm_applied] = (pred_toks != labels[is_mlm_applied]) # (B,L)

    disc_logits = self.discriminator(generated, attention_mask, token_type_ids)[0] # (B, L)

    return mlm_gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied

  def _get_pad_mask_and_token_type(self, input_ids, sentA_lenths):
    """
    Only cost you about 500 µs for (128, 128) on GPU, but so that your dataset won't need to save attention_mask and token_type_ids and won't be unnecessarily large, thus, prevent cpu processes loading batches from consuming lots of cpu memory and slow down the machine. 
    """
    attention_mask = input_ids != self.tokenizer.pad_token_id
    seq_len = input_ids.shape[1]
    token_type_ids = torch.tensor([ ([0]*len + [1]*(seq_len-len)) for len in sentA_lenths.tolist()],  
                                  device=input_ids.device)
    return attention_mask, token_type_ids

  def sample(self, logits):
    "Reimplement gumbel softmax cuz there is a bug in torch.nn.functional.gumbel_softmax when fp16 (https://github.com/pytorch/pytorch/issues/41663). Gumbel softmax is equal to what official ELECTRA code do, standard gumbel dist. = -ln(-ln(standard uniform dist.))"
    if self.sampling == 'fp32_gumbel':
      return (logits.float() + self.gumbel_dist.sample(logits.shape)).argmax(dim=-1)
    elif self.sampling == 'fp16_gumbel': # 5.06 ms
      return (logits + self.gumbel_dist.sample(logits.shape)).argmax(dim=-1)
    elif self.sampling == 'multinomial': # 2.X ms
      return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze()

class RTDLoss():
  def __init__(self, loss_weights=(1.0, 50.0), gen_label_smooth=False, disc_label_smooth=False):
    self.loss_weights = loss_weights
    self.gen_loss_fc = LabelSmoothingCrossEntropyFlat(eps=gen_label_smooth) if gen_label_smooth else CrossEntropyLossFlat()
    self.disc_loss_fc = nn.BCEWithLogitsLoss()
    self.disc_label_smooth = disc_label_smooth
    
    
  def __call__(self, pred, targ_ids):
    mlm_gen_logits, generated, disc_logits, is_replaced, non_pad, is_mlm_applied = pred
    gen_loss = self.gen_loss_fc(mlm_gen_logits.float(), targ_ids[is_mlm_applied])
    disc_logits = disc_logits.masked_select(non_pad) # -> 1d tensor
    is_replaced = is_replaced.masked_select(non_pad) # -> 1d tensor
    if self.disc_label_smooth:
      is_replaced = is_replaced.float().masked_fill(~is_replaced, self.disc_label_smooth)
    disc_loss = self.disc_loss_fc(disc_logits.float(), is_replaced.float())
    return gen_loss * self.loss_weights[0] + disc_loss * self.loss_weights[1]  
