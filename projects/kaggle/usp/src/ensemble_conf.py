v1 = 0
mns1 = [

]

v2 = 0
mns2 = [    
]

v3 = 4
mns3 = [    
#  'deberta-v3.lr_decay_power-1',
#  'deberta-v2-xlarge.1',
#  'deberta-xlarge.1',
  # 'deberta-v3.method-pearson.1',
  # 'deberta-v3.method-pearson.1',
  # 'deberta-v3.method-soft_binary.1',
  # 'deberta-v3.method-mse.1',
  'deberta-v3',
  'deberta-v3.pooling-latt',
  # 'deberta-v3.pooling-att',
  'deberta-v3.method-soft_binary',
  'patent',
  'patent.pooling-latt',
  # 'deberta-v3.method-mse',
    # 'deberta-v3.lower',
    # 'deberta-v3.pooling-latt',
  # 'deberta-v3-base',
]

# v3 = 9
# mns3 = [    
#         'deberta-v3-large.flag-best.scheduler-bert.lr_decay_power-1',
#         'patent.flag-best.shuffle_targets',
#         'simcse-patent.flag-best.shuffle_targets',
#         # 'deberta-v3-large.flag-best.shuffle_targets.pooling-cls',
#         'electra-squad.scheduler-cosine.lr-2e-5',
#         # # 'patent.scheduler-cosine.pooling-cls',
#         # # 'deberta-v3.binary_loss_rate-0.05',
#         # #'deberta-v3.pooling-mean.0616',
#         # 'deberta',
#         # 'deberta-mnli.0617',
#         'deberta-v3.method-pearson2.0617',
# ]

v3 = 10
mns3 = [
  # 'deberta-v3.rnn.x',
  # 'patent.x',
  # 'deberta-v3.use_sector.x',
  # 'deberta-v3.rnn_double_dim.x',
  # 'deberta-v3.rnn_type-GRU.x',
  # 'deberta-v3.pooling-mean.x',
  # 'deberta-v3.pooling-latt,mean.x',
  # 'deberta-v3.context_key-sector.x',
  # 'deberta-v3.dynamic_aug_seed.x',
  # 'patent.dynamic_aug_seed.aug_seed-1000.x',
  # 'electra-squad.x',
  # 'simcse-patent.x',
  # 'deberta.x',
  
  'deberta-v3.lr-2e-5.x',
  'patent.rnn_double_dim.x',
  'electra-squad.x',
  'simcse-patent.rnn_double_dim.x',
  'deberta-v3.lr-2e-5.context_key-sector.x',
  'patent.rnn_double_dim.context_key-sector.x',
  'electra-squad.context_key-sector.x',
  ## 'simcse-patent.rnn_double_dim.context_key-sector.x',
  # 'deberta-v3.lr-2e-5.fold_seed-1025.x',
  # 'patent.rnn_double_dim.fold_seed-1026.x',
  # 'electra-squad.fold_seed-1027.x',
  # 'simcse-patent.rnn_double_dim.fold_seed-1028.x',
  # 'deberta-v3.lr-2e-5.context_key-sector.fold_seed-100.x',
  # 'patent.rnn_double_dim.context_key-sector.fold_seed-101.x',
  #'electra-squad.context_key-sector.fold_seed-102.x',
]
mns = mns1 + mns2 + mns3
v = v3

weights_dict = {}

weights_dict = {
  'deberta-v3.rnn.x': 1,
  'patent.x': 0.4,
  'simcse-patent.x': 0.2,
  'deberta-v3.dynamic_aug_seed.x': 1,
  'patent.dynamic_aug_seed.aug_seed-1000.x': 0.4,
  'deberta-v3.use_sector.x': 0.8,
  'deberta-v3.context_key-sector.x': 0.8,
  'deberta-v3.rnn_double_dim.x': 1.,
  'deberta-v3.rnn_type-GRU.x': 0.8,
  'deberta-v3.pooling-mean.x': 0.8,
  'deberta-v3.pooling-latt,mean.x': 0.8,
  'deberta-v3.lr-2e-5.x': 0.6,
  'patent.rnn_double_dim.x': 0.3,
  'electra-squad.x': 0.2,
  'simcse-patent.rnn_double_dim.x': 0.2,
  'deberta-v3.lr-2e-5.context_key-sector.x': 0.4,  
  'patent.rnn_double_dim.context_key-sector.x': 0.1,
  'electra-squad.context_key-sector.x': 0.1,
  # 'simcse-patent.rnn_double_dim.context_key-sector.x': 0.02,
}

# weights_dict = {
#   'deberta-v3.rnn.x': 1.2,
#   'patent.x': 0.4,
#   'electra-squad.x': 0.3,
#   'simcse-patent.x': 0.2,
# }

# weights_dict = {
#   'deberta-v3.rnn.x': 1.,
#   'patent.x': 0.3,
#   'electra-squad.x': 0.2,
#   'simcse-patent.x': 0.1,
# }


# weights = [1] * len(mns)
def get_weight(x):
  return weights_dict.get(x, 1)
  # return weights_dict[x]
  return 1

if all(x in weights_dict for x in mns):
  weights = [weights_dict[x] for x in mns]
else:
  weights = [get_weight(x) for x in mns]
weights.extend([1] * 100)
ic(list(zip(mns, weights)), len(mns))

SAVE_PRED = 0
