#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2022-05-11 11:12:25.201944
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 

def get_callbacks(model):
  callbacks = []

  if FLAGS.torch:
    if FLAGS.freeze_epochs > 0:
      callbacks.append(lele.FreezeCallback(model.backbone, FLAGS.freeze_epochs))
  ic(callbacks)
  return callbacks
