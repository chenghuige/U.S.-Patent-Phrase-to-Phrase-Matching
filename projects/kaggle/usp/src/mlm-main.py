#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   main.py
#        \author   chenghuige  
#          \date   2022-02-10 07:02:54.233162
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_DEBUG"] = 'WARNING'


from tensorflow import keras
from torch.utils.data import DataLoader

from transformers import (
  AutoModelForMaskedLM,
  DataCollatorForLanguageModeling,
)
import datasets

import gezi
from gezi import tqdm
logging = gezi.logging
import melt as mt
import lele

import src
import src.eval as ev
from src import config
from src.config import *
from src.preprocess import *
from src import util

import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

def main(_):
  timer = gezi.Timer()
  fit = mt.fit  
  
  rank = gezi.get('RANK', 0)
  if rank != 0:
    ic.disable()
  
  FLAGS.caching = True
  FLAGS.hug_inputs = True
  FLAGS.continue_pretrain = False
  FLAGS.online = True
  FLAGS.ep = FLAGS.ep or 8
  FLAGS.bs = FLAGS.bs or 32
  config.init()
  FLAGS.gpus = -1
  ic(FLAGS.gpus, FLAGS.bs)
  backbone = FLAGS.backbone.split('/')[-1]
  # notice for lr, all other converge using 5e-5 except deberta-xlarge which also use bs 8 instead of 16 due to OOM on 4 A100 GPUs, so deberta-xlarge using 2.5e-5
  FLAGS.lr_decay_power = 1.
  
  FLAGS.mn = backbone
  FLAGS.model_dir = f'{FLAGS.root}/pretrain/base'
  FLAGS.sie = 1
  FLAGS.awp_train = False
  FLAGS.ema_train = False
  
  mt.init()
  
  ic(FLAGS.model_dir, FLAGS.mn, FLAGS.backbone, backbone)
  tokenizer = get_tokenizer(FLAGS.backbone)
  ic(tokenizer)
  
  text_column_name = 'text'
  
  dfs = []
  # ifile = f'{FLAGS.root}/pppm-abstract/pppm_abstract.csv'
  ifile = f'{FLAGS.root}/sampled-patent-titles/sampled-patent-titles.csv'
  df = pd.read_csv(ifile)
  # df[text_column_name] = df['abstract']
  df[text_column_name] = df['title'] + '[SEP]' + df['abstract']
  df['text_len'] = df[text_column_name].apply(lambda x: len(x) if isinstance(x, str) else 0)
  df = df[df.text_len > 5]
  df = df[[text_column_name]]
  dfs.append(df)
  
  df = pd.concat(dfs)
  ds = datasets.Dataset.from_pandas(df)
  
  ic(ds, ds[-1])
  num_proc = 32 if FLAGS.pymp else 1
  gezi.try_mkdir(f'{FLAGS.root}/cache')
  
  def preprocess(text, method=None):
    return text
  
  ds = ds.map(lambda example: {text_column_name: preprocess(example[text_column_name])}, 
              remove_columns=ds.column_names,
              batched=False, 
              num_proc=num_proc, 
              )
  ic(ds, ds[-1])

  def tokenize_function(examples):
      # Remove empty lines
      examples[text_column_name] = [
          line for line in examples[text_column_name] if line and len(line) > 0 and not line.isspace()
      ]
      return tokenizer(
          examples[text_column_name],
          padding=False,
          truncation=True,
          max_length=256,
          return_special_tokens_mask=True,
      )

  ds = ds.map(
      tokenize_function,
      batched=True,
      num_proc=num_proc,
      remove_columns=[text_column_name],
      desc="Running tokenizer on dataset line_by_line",
  )
  ic(ds)
  
  collate_fn = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=None,
    )

  # collate_fn=gezi.DictPadCollate()
  sampler = lele.get_sampler(ds, shuffle=True)
  kwargs = {'num_workers': 8, 'pin_memory': True, 'persistent_workers': True, 'collate_fn':collate_fn} 
  dl = torch.utils.data.DataLoader(ds, batch_size=gezi.batch_size(), sampler=sampler, **kwargs)
  
  model = AutoModelForMaskedLM.from_pretrained(FLAGS.backbone)
  model.resize_token_embeddings(len(tokenizer)) 
  if FLAGS.opt_fused:
    lele.replace_with_fused_layernorm(model)
  
  fit(model,  
      dataset=dl,
      opt_params=lele.get_opt_params(model, weight_decay=FLAGS.weight_decay),
    ) 

  if rank == 0:          
    tokenizer.save_pretrained(FLAGS.model_dir)
    model.save_pretrained(FLAGS.model_dir)
  
if __name__ == '__main__':
  app.run(main)  
