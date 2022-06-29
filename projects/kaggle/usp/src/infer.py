#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   infer.py
#        \author   chenghuige  
#          \date   2022-05-15 07:00:15.332073
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
if os.path.exists('/kaggle'):
  sys.path.append('/kaggle/input/pikachu/utils')
  sys.path.append('/kaggle/input/pikachu/third')
  sys.path.append('.')
else:
  sys.path.append('..')
  sys.path.append('../../../../utils')
  sys.path.append('../../../../third')

from gezi.common import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_DEBUG"] = 'WARNING'

from src.config import *
from src.preprocess import *
from src.postprocess import *

def main(argv):
  model_dir = sys.argv[1]
  eval_bs = int(sys.argv[2])
  out_file = sys.argv[3]
  
  ic(model_dir, eval_bs, out_file)
  gezi.init_flags()
  gezi.restore_configs(model_dir)
  ic(FLAGS.model_dir)
  FLAGS.train_allnew = False
  FLAGS.bs = eval_bs
  FLAGS.eval_bs = eval_bs
  FLAGS.grad_acc = 1
  mt.init()
  FLAGS.pymp = False
  FLAGS.num_workers = 1
  FLAGS.pin_memory = False
  FLAGS.persistent_workers = False
  # FLAGS.fold = 0

  if len(sys.argv) > 4:  
   FLAGS.static_inputs_len = bool(sys.argv[4])
   FLAGS.hack_infer = bool(sys.argv[5])
   
  FLAGS.bs = eval_bs
  FLAGS.eval_bs = eval_bs
  FLAGS.grad_acc = 1
  
  if gezi.in_kaggle():
    backbone = FLAGS.backbone.split('/')[-1].replace('_', '-')
    FLAGS.backbone = '../input/' + backbone
  
  FLAGS.work_mode = 'test'
  FLAGS.model_dir = model_dir
  test_ds = get_dataloaders(test_only=True)
  # ic(test_ds.dataset[0])
  ic(FLAGS.backbone, FLAGS.model_dir, os.path.exists(f'{FLAGS.model_dir}/model.pt'))
  display(pd.read_csv(f'{model_dir}/metrics.csv'))
  
  if not FLAGS.torch:
    from src.tf.model import Model
  else:
    from src.torch.model import Model
  model = Model()
  
  ic(gezi.get_mem_gb())
  gezi.load_weights(model, model_dir)
  ic(gezi.get_mem_gb())

  x = lele.predict(model, test_ds, out_keys=['id', 'label', 'anchor', 'target', 'context'])
  
  x['pred'] = to_pred(x['pred'])
  #m = gezi.get('infer_dict')
  #x['id'] = m['id']
  ic(x)
  gezi.save(x, out_file)

if __name__ == '__main__':
  app.run(main)  
  
