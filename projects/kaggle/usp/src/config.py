#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   config.py
#        \author   chenghuige  
#          \date   2022-05-11 11:18:18.068436
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy import clip

from regex import F

from gezi.common import * 

RUN_VERSION = '10'
PREFIX = '.y'
MODEL_NAME = 'usp'

flags.DEFINE_integer('fold_seed', 1024, '')
flags.DEFINE_integer('fold_seed2', None, '')
flags.DEFINE_integer('valid_fold_idx', 1, '')
flags.DEFINE_bool('tf', False, '')
flags.DEFINE_string('root', '../input/us-patent-phrase-to-phrase-matching', '')
flags.DEFINE_string('hug', 'deberta-v3', '')
flags.DEFINE_string('backbone', None, '')
flags.DEFINE_bool('lower', False, '')
flags.DEFINE_integer('max_len', 192, '')
flags.DEFINE_bool('static_inputs_len', False, '')
flags.DEFINE_bool('token_types', False, '')
flags.DEFINE_bool('group_context', True, '')
flags.DEFINE_bool('remove_dup_target', True, '')
flags.DEFINE_bool('remove_self_target', True, '')
flags.DEFINE_bool('remove_self_anchor', False, '')
flags.DEFINE_bool('use_code', False, '')
flags.DEFINE_bool('use_sector', False, '')
flags.DEFINE_bool('use_anchor', False, '')
flags.DEFINE_string('stratify_key', 'cat', '')
flags.DEFINE_bool('split_cpc', False, '')
flags.DEFINE_bool('use_targets_count', False, '')
flags.DEFINE_bool('add_context', False, '')
flags.DEFINE_string('context_key', 'context', '')
flags.DEFINE_integer('targets_idx', 3, '')
flags.DEFINE_bool('trans_chem', False, '')

flags.DEFINE_string('method', 'pearson', 'soft_binary, cls, mse, pearson, pearson2')
flags.DEFINE_bool('cluster_score', False, '')
flags.DEFINE_bool('transform_score', False, '')
flags.DEFINE_float('mse_loss_rate', 0.0, '')
flags.DEFINE_float('binary_loss_rate', 0.0, '')
flags.DEFINE_float('mean_score_rate', 0.0, '')

flags.DEFINE_bool('multi_lr', True, '')
flags.DEFINE_bool('continue_pretrain', False, '')
flags.DEFINE_float('base_lr', None, '1e-3 or 5e-4')
flags.DEFINE_bool('weight_decay', True, '')

flags.DEFINE_bool('awp', True, '')
flags.DEFINE_string('pooling', 'latt', 'latt a bit better then cls')
flags.DEFINE_bool('pooling_mask', True, '')
flags.DEFINE_bool('concat_last', False, '')
flags.DEFINE_float('drop_rate', 0, '')

flags.DEFINE_float('aug_prob', 0., '')
flags.DEFINE_float('mask_prob', 0.1, '')
flags.DEFINE_integer('aug_seed', 2048, '')
flags.DEFINE_bool('dynamic_aug_seed', False, '')
flags.DEFINE_bool('mask_self', True, '')
flags.DEFINE_bool('shuffle_targets', True, '')
flags.DEFINE_bool('shuffle_valid_targets', True, '')
flags.DEFINE_float('shuffle_context_prob', 0., '')
flags.DEFINE_bool('sample_targets', False, '')

flags.DEFINE_bool('two_tower', False, '')
flags.DEFINE_bool('layer_norm', False, '')
flags.DEFINE_bool('sigmoid', False, '')
flags.DEFINE_bool('rnn', True, '')
flags.DEFINE_integer('rnn_layers', 1, '')
flags.DEFINE_bool('rnn_bi', True, '')
flags.DEFINE_float('rnn_dropout', 0.1, '')
flags.DEFINE_string('rnn_type', 'LSTM', '')
flags.DEFINE_bool('rnn_double_dim', False, '')
flags.DEFINE_bool('freeze_emb', True, '')
flags.DEFINE_bool('prompt', False, '')

flags.DEFINE_bool('ensemble_select', False, '')

hugs = {
  'tiny': 'roberta-base',
  'roberta': 'roberta-large', # ok
  'mid': 'roberta-large',
  'electra': 'google/electra-large-discriminator', # ok electra版本效果不错 甚至好于roberta512 但是某些fold有很大概率loss不下降崩溃 没找到原因 但是online全量训练应该还好可以看train自己loss是否正常 在线提交融合结果也还ok
  'electra-base': 'google/electra-base-discriminator',
  'bart': 'facebook/bart-large', # okk
  'bart-base': 'facebook/bart-base',
  'large-qa': 'allenai/longformer-large-4096-finetuned-triviaqa',
  'roberta-base': 'roberta-base',
  'ro': 'roberta-base',
  'fast': 'roberta-base',
  'roberta-large': 'roberta-large',
  'xlnet-large': 'xlnet-large-cased',
  'xlnet': 'xlnet-large-cased',
  'robertam': 'roberta-large-mnli',
  'xlm-roberta': 'xlm-roberta-large',
  'deberta-mnli': 'khalidalt/DeBERTa-v3-large-mnli',
  'info-xlm': 'microsoft/infoxlm-large',
  'robertas': 'deepset/roberta-large-squad2',
  'roberta-squad': 'deepset/roberta-large-squad2',
  'reformer': 'google/reformer-enwik8', #fail
  'roformer': 'junnyu/roformer_chinese_base',
  'span': 'SpanBERT/spanbert-large-cased', # need --br='[SEP]'
  'gpt2': 'gpt2-large', # not well
  'berts': 'phiyodr/bart-large-finetuned-squad2',
  'barts': 'phiyodr/bart-large-finetuned-squad2',
  'bart-squad': 'phiyodr/bart-large-finetuned-squad2', # this is file
  'albert': 'albert-large-v2',
  'bert-cased': 'bert-large-cased',
  'bert-uncased': 'bert-large-uncased',
  'bert-squad': 'deepset/bert-large-uncased-whole-word-masking-squad2', # fail
  't5': 't5-large',
  'base-squad': 'valhalla/longformer-base-4096-finetuned-squadv1',
  'albert-squad': 'mfeb/albert-xxlarge-v2-squad2',
  'electra-squad': 'ahotrod/electra_large_discriminator_squad2_512',
  'deberta': 'microsoft/deberta-large',
  'deberta-base': 'microsoft/deberta-base',
  'deberta-xl': 'microsoft/deberta-xlarge',
  'deberta-xlarge': 'microsoft/deberta-xlarge',
  'deberta-v2': 'microsoft/deberta-v2-xlarge', # v2 v3 all has problem... of tokenizer no fast/non python version (hack now)
  'deberta-v2-xlarge': 'microsoft/deberta-v2-xlarge', # v2 v3 all has problem... of tokenizer no fast/non python version (hack now)
  'deberta-v2-xxlarge': 'microsoft/deberta-v2-xxlarge',
  'deberta-v3': 'microsoft/deberta-v3-large', 
  'deberta-v3-large': 'microsoft/deberta-v3-large', 
  'deberta-v3-base': 'microsoft/deberta-v3-base',
  'deberta-v3-mnli': 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
  'patent': 'anferico/bert-for-patents',
  'patent-cpc': 'bradgrimm/patent-cpc-predictor',
  'bart-patent': 'Pyke/bart-finetuned-with-patent',
  'peasus-patent': 'google/pegasus-big_patent',
  'coco': 'microsoft/cocolm-large',
  'psbert': 'AI-Growth-Lab/PatentSBERTa',
  'simcse-patent': 'Yanhao/simcse-bert-for-patent',
  'simcse-roberta': 'Yanhao/simcse-roberta-large',
  'patent-deberta': 'tanapatentlm/patentdeberta_base_total_512',
  'scibert-patent': 'kaesve/SciBERT_patent_reference_extraction',
  'bigpatent': 'hyunwoongko/ctrlsum-bigpatent',
  'xlm-roberta-large': 'xlm-roberta-large',
  'xlm': 'xlm-roberta-large',
  'unilm': 'microsoft/unilm-large-cased',
  'erine': 'nghuyong/ernie-2.0-large-en',
  'funnel': 'funnel-transformer/large',
}

def get_backbone(backbone, hug):
  backbone = backbone or hugs.get(hug, hug)
  ic(backbone)
  
  backbone_ = backbone.split('/')[-1]
  if FLAGS.continue_pretrain:
    backbone_path = f'{FLAGS.root}/pretrain/{backbone_}'
    if os.path.exists(f'{backbone_path}/config.json'):
      backbone = backbone_path
      return backbone
  backbone_path = f'{FLAGS.root}/{backbone_}'
  if os.path.exists(f'{backbone_path}/config.json'):
    backbone = backbone_path
    return backbone
    
  return backbone

def get_records_name():
  records_name = FLAGS.backbone.split('/')[-1]
  return records_name

def show():
  ic(FLAGS.backbone, 
     FLAGS.lower, 
     FLAGS.method,
     FLAGS.pooling, 
     FLAGS.concat_last, 
     FLAGS.remove_self_target,
     FLAGS.shuffle_targets, 
     FLAGS.shuffle_valid_targets,
     FLAGS.rnn,
     FLAGS.prompt)

def config_train():
  lr = 3e-5
  bs = 128
  
  if 'base' in FLAGS.backbone:
    bs *= 2
  
  # FLAGS.lr_decay_power = 0.5
  FLAGS.lr_decay_power = 2
  
  if 'xlarge' in FLAGS.hug:
    lr = 1e-5
    # bs = 64
    FLAGS.grad_acc *= 4
    
  ## electra-squad better then electra but still a bit unstable might need 2e-5
  ## dev3 tested 2e-5 better, for others like bert-for-patents use 3e-5
  ## well in 1 run 2e-5 on folds_seed 1024,1025 both much better cv then 3e-5
  ## but for single model 5 folds, seems still 3e-5 a litte better on PB (both 8761) and much better on LB(8615 3e-5 with 8604 2e-5)
  ## so basicly for most bert models 3e-5 is still the best finetune choice
  if any(name in FLAGS.hug for name in ['electra', 'deberta']):
    lr = 2e-5
  
  # if 'deberta-v2-xlarge' in FLAGS.hug:
  #   lr = 1e-5
  #   bs = 64
  #   # FLAGS.lr_decay_power = 2
  
  FLAGS.lr = FLAGS.lr or lr
  FLAGS.clip_gradients = 1
  
  # versy sensitive to lr ie, for roformer v2 large, 5e-5 + 5e-4 will not converge but 5e-5 + 1e-4 will
  # also TODO with 1e-4 + 1e-3 lr, opt_fused very interesting download with roformer v2 large layer norm .. random init due to key miss in checkpoint will converge
  # but if save_pretrained then reload will not, why ?
  if FLAGS.multi_lr:
    # base_lr = FLAGS.lr * 10.
    base_lr = 1e-3
    FLAGS.base_lr = FLAGS.base_lr or base_lr
    
  # FLAGS.loss_scale = 100
  FLAGS.bs = FLAGS.bs or bs
  FLAGS.eval_bs = FLAGS.eval_bs or FLAGS.bs * 2
  
  if FLAGS.max_len > 160 and FLAGS.bs >= 128:
    FLAGS.grad_acc *= 2
  
    if FLAGS.rdrop_rate > 0:
      FLAGS.grad_acc *= 2
      
    if FLAGS.prompt and 'deberta' in FLAGS.backbone:
      FLAGS.grad_acc *= 2
  
  ep = 5
  FLAGS.ep = FLAGS.ep or ep
  
  # change from adamw back to adam
  optimizer = 'adamw' 
  FLAGS.optimizer = FLAGS.optimizer or optimizer
  FLAGS.opt_eps = 1e-7
  
  scheduler = 'bert' if 'deberta' in FLAGS.backbone else 'cosine'
  FLAGS.scheduler = FLAGS.scheduler or scheduler

  if FLAGS.tf_dataset:
    records_pattern = f'{FLAGS.root}/{FLAGS.records_name}/{get_records_name()}/train/*.tfrec'
    ic(records_pattern)
    files = gezi.list_files(records_pattern) 
    FLAGS.valid_files = [x for x in files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds == FLAGS.fold]
    if FLAGS.online:
      FLAGS.train_files = files
    else:
      FLAGS.train_files = [x for x in files if x not in FLAGS.valid_files]
    
    ic(FLAGS.train_files[:2], FLAGS.valid_files[:2])
    
    if not FLAGS.train_files:
      FLAGS.tf_dataset = False
    
  if FLAGS.dynamic_aug_seed:
    if FLAGS.online:
      FLAGS.aug_seed -= 1
    else:
      FLAGS.aug_seed += FLAGS.fold    
  
def config_model():
  FLAGS.backbone = get_backbone(FLAGS.backbone, FLAGS.hug)
 
  awp_train = True
  FLAGS.awp_train = awp_train if FLAGS.awp_train is None else FLAGS.awp_train
  # if not any(x in FLAGS.hug for x in ['xlarge', 'roberta', 'electra']):
  #   FLAGS.rdrop_rate = 0.1
    
  if any(x in FLAGS.hug for x in ['roberta', 'xlnet', 'patent']):
    FLAGS.find_unused_parameters = True
  
  if FLAGS.concat_last:
    FLAGS.find_unused_parameters = True
  
  if FLAGS.prompt:
    FLAGS.rnn = False  
  
def init():
  config_model()

  folds = 5 
  FLAGS.folds = folds if FLAGS.folds is None else FLAGS.folds
  FLAGS.fold = FLAGS.fold or 0
  # FLAGS.show_keys = ['score']

  FLAGS.buffer_size = 20000
  FLAGS.static_input = True
  FLAGS.cache_valid = True
  FLAGS.async_eval = True
  FLAGS.async_eval_last = True if not FLAGS.pymp else False
  FLAGS.async_valid = False
  
  # FLAGS.find_unused_parameters=True
  
  if not FLAGS.tf:
    # make torch by default
    FLAGS.torch = True
  else:
    FLAGS.torch = False
  
  ic(FLAGS.torch, FLAGS.torch_only, FLAGS.tf_dataset)

  
  FLAGS.run_version = FLAGS.run_version or RUN_VERSION
      
  if FLAGS.online:
    FLAGS.allow_train_valid = True
    if FLAGS.fold_seed2 is None:
      FLAGS.nvs = 1
    # assert FLAGS.fold == 0
    # if FLAGS.fold != 0:
    #   ic(FLAGS.fold)
    #   exit(0)
     
  if FLAGS.log_all_folds or FLAGS.fold == 0:
    FLAGS.wandb = True
    FLAGS.wandb_project = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  FLAGS.write_summary = True
  
  FLAGS.run_version += f'/{FLAGS.fold}'
  
  pres = ['offline', 'online']
  pre = pres[int(FLAGS.online)]
  model_dir = f'../working/{pre}/{FLAGS.run_version}/model'  
  FLAGS.model_dir = FLAGS.model_dir or model_dir
  if FLAGS.mn == 'model':
    FLAGS.mn = ''
  if not FLAGS.mn:  
    if FLAGS.hug:
      model_name = FLAGS.hug
    FLAGS.mn = model_name
    if FLAGS.tf:
      FLAGS.mn = f'tf.{FLAGS.mn}'
      
    mt.model_name_from_args(ignores=['tf', 'hug', 'test_file', 'static_inputs_len'])
    FLAGS.mn += PREFIX
  
  config_train()
  
  if not FLAGS.online:
    FLAGS.nvs = FLAGS.nvs or FLAGS.ep
    # FLAGS.vie = 1
    # if FLAGS.method == 'gp':
    #   FLAGS.nvs = 1
    
  FLAGS.write_valid_final = True
  FLAGS.save_model = False
  FLAGS.sie = 1e10  
  # FLAGS.sie = FLAGS.ep
  
  show()
  
