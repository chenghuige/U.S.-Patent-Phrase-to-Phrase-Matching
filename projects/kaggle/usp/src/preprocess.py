#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   preprocess.py
#        \author   chenghuige
#          \date   2022-05-11 11:12:36.045278
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
from src.config import *
from sklearn.preprocessing import QuantileTransformer


def get_cpc_texts():
  contexts = []
  pattern = '[A-Z]\d+'
  for file_name in os.listdir('../input/cpc-data/CPCSchemeXML202105'):
    result = re.findall(pattern, file_name)
    if result:
      contexts.append(result)
  contexts = sorted(set(sum(contexts, [])))
  results = {}

  for cpc in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y']:
    with open(
        f'../input/cpc-data/CPCTitleList202202/cpc-section-{cpc}_20220201.txt'
    ) as f:
      s = f.read()
    pattern = f'{cpc}\t\t.+'
    result = re.findall(pattern, s)
    pattern = "^" + pattern[:-2]
    cpc_result = re.sub(pattern, "", result[0])
    for context in [c for c in contexts if c[0] == cpc]:
      pattern = f'{context}\t\t.+'
      result = re.findall(pattern, s)
      pattern = "^" + pattern[:-2]
      sep = ". " if not FLAGS.split_cpc else "[SEP]"
      results[context] = cpc_result + sep + re.sub(pattern, "", result[0])
      if FLAGS.lower:
        results[context] = results[context].lower()

  # for cpc in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y']:
  #   with open(
  #       f'../input/cpc-data/CPCTitleList202202/cpc-section-{cpc}_20220201.txt'
  #   ) as f:
  #     s = f.read()
  #   pattern = f'{cpc}\t\t.+'
  #   result = re.findall(pattern, s)
  #   cpc_result = result[0].lstrip(pattern)
  #   for context in [c for c in contexts if c[0] == cpc]:
  #     pattern = f'{context}\t\t.+'
  #     result = re.findall(pattern, s)
  #     results[context] = cpc_result + ". " + result[0].lstrip(pattern)

  return results

# with open('../input/periodictable/periodic_table.p', 'rb') as fin:
#   import pickle
#   per_table = pickle.load(fin)

# def atoms_to_str(atoms):
#   return ' '.join([per_table.get(x.lower(), '') for x in atoms])

# def parse_formula(text):
#   import chemparse
#   tokenized = text.split(' ')

#   results = []

#   for tok in tokenized:
#     atoms = chemparse.parse_formula(tok).keys()
#     formula = atoms_to_str(atoms)
#     if len(formula) < 2 or len(tok) < 3:
#       results.append(tok)
#     else:
#       try:
#         f = Formula(tok.upper())
#         atoms = f.atoms
#         formula = ' '.join([x.name.lower() for x in atoms])
#       except Exception as e:
#         pass

#       results.append(formula)

#   return ' '.join(results)

# def parse_df_formulas(df):
#   df = df.copy()
#   df.loc[:, 'target'] = df.target.apply(parse_formula)
#   return df


def set_fold(df):
  intersect_words = set(df.target) & set(df.anchor)
  d = df[df.target.isin(intersect_words)]
  d = d[d.anchor != d.target]
  anchors = gezi.unique_list(d.anchor.values)
  group_dict = {}
  idx = 0
  for row in d.itertuples():
    if (row.anchor not in group_dict) and (row.target not in group_dict):
      group_dict[row.anchor] = str(idx)
      group_dict[row.target] = str(idx)
      idx += 1
    elif row.anchor not in group_dict:
      group_dict[row.anchor] = group_dict[row.target]
    elif row.target not in group_dict:
      group_dict[row.target] = group_dict[row.anchor]
  df['group'] = df.anchor.apply(lambda x: group_dict.get(x, x))
  gezi.set_fold_worker(df,
                       FLAGS.folds,
                       80,
                       group_key='group',
                       stratify_key=FLAGS.stratify_key,
                       seed=FLAGS.fold_seed)
  if FLAGS.fold_seed2 is not None:
    # for fold_seed 1024 seems fold 1 better represent online score
    df_valid = df[df.fold == FLAGS.valid_fold_idx]
    df_train = df[df.fold != FLAGS.valid_fold_idx]
    df_valid.fold = -1
    # use first 5 folds of 10 folds
    gezi.set_fold_worker(df_train,
                         FLAGS.folds * 2,
                         80,
                         group_key='group',
                         stratify_key='cat',
                         seed=FLAGS.fold_seed2)
    df = pd.concat([df_valid, df_train])
  return df


cpc_texts = None
pair_scores = {}


def prepare(df, infer=False):
  # if FLAGS.trans_chem:
  #   df = parse_df_formulas(df)
  if not infer and FLAGS.sample_targets:
    for row in df.itertuples():
      pair_scores[f'{row.anchor}\t{row.target}'] = row.score
  
  if infer:
    df_train = get_df('train')
  
  if not FLAGS.group_context:
    df_ = df
    if FLAGS.remove_dup_target:
      df_ = df[df.anchor != df.target]
    targets = df_.groupby('anchor')['target'].apply(list).reset_index(
        name='targets')
  else:
    # according to https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332234
    # there are duplicate anchor for train/test so merge it on test time
    df_ = df if not infer else pd.concat([df_train, df])
    targets = df_.groupby(['anchor', FLAGS.context_key
                         ])['target'].apply(list).reset_index(name='targets')
    if FLAGS.use_sector:
      targets2 = df_.groupby(
          ['anchor',
           'sector'])['target'].apply(list).reset_index(name='targets2')
    if FLAGS.use_anchor:
      targets3 = df_.groupby(
          ['anchor'])['target'].apply(list).reset_index(name='targets3')

  df = df.merge(targets, on=['anchor', FLAGS.context_key])
  if FLAGS.use_sector:
    df = df.merge(targets2, on=['anchor', 'sector'])
  if FLAGS.use_anchor:
    df = df.merge(targets3, on=['anchor'])

  if FLAGS.remove_self_target:
    df['targets_list'] = [
        [x for x in row.targets if x != row.target] for row in df.itertuples()
    ]
    if FLAGS.use_sector:
      df['targets_list2'] = [[
          x for x in row.targets2 if (x != row.target and x not in row.targets)
      ] for row in df.itertuples()]
    if FLAGS.use_anchor:
      df['targets_list3'] = [[
          x for x in row.targets3 if (x != row.target and x not in row.targets2)
      ] for row in df.itertuples()]
  if FLAGS.remove_self_anchor:
    df['targets_list'] = [
        [x for x in row.targets if x != row.anchor] for row in df.itertuples()
    ]
  df['targets'] = df['targets_list'].apply(lambda l: '; '.join(l))
  if FLAGS.use_sector:
    df['targets2'] = df['targets_list2'].apply(lambda l: '; '.join(l))
  if FLAGS.use_anchor:
    df['targets3'] = df['targets_list3'].apply(lambda l: '; '.join(l))

  if FLAGS.mean_score_rate > 0:
    if not infer:
      anchor_scores = df.groupby([
          'anchor', 'context'
      ])['score'].apply(lambda l: np.asarray(list(l)).mean()).reset_index(
          name='mean_score')
      df = df.merge(anchor_scores, on=['anchor', 'context'])
    else:
      df['mean_score'] = 0.

  # if not FLAGS.use_code:
  global cpc_texts
  if cpc_texts is None:
    cpc_texts = get_cpc_texts()
  df['context_text'] = df['context'].map(cpc_texts)
  # else:
  #   code_df = pd.read_csv('../input/cpc-codes/titles.csv')[['code', 'title']].rename(columns={'title': 'context_text'})
  #   if FLAGS.lower:
  #     code_df['context_text'] = code_df.context_text.apply(lambda x: x.lower())
  #   df = pd.merge(df, code_df, left_on='context', right_on='code', how='left')
  # if FLAGS.use_sector:
  #   code_df = pd.read_csv('../input/cpc-codes/titles.csv')[['code', 'title']].rename(columns={'title': 'sector_text'})
  #   if FLAGS.lower:
  #     code_df['sector_text'] = code_df.sector_text.apply(lambda x: x.lower())
  #   # code_df.rename(columns={'context_text': 'sector_text'}, inplace=True)
  #   df = pd.merge(df, code_df, left_on='sector', right_on='code', how='left')
  #   df['text'] = df['anchor'] + '[SEP]' + df['target'] + '[SEP]'  + df['sector_text'] + '[SEP]' + df['context_text'] + '[SEP]' + df['targets']
  # else:
  if not FLAGS.use_sector:
    df['text'] = df['anchor'] + '[SEP]' + df['target'] + '[SEP]' + df[
        'context_text'] + '[SEP]' + df['targets']
  else:
    if not FLAGS.use_anchor:
      df['text'] = df['anchor'] + '[SEP]' + df['target'] + '[SEP]' + df[
          'context_text'] + '[SEP]' + df['targets'] + '[SEP]' + df['targets2']
    else:
      df['text'] = df['anchor'] + '[SEP]' + df['target'] + '[SEP]' + df[
          'context_text'] + '[SEP]' + df['targets'] + '[SEP]' + df[
              'targets2'] + '[SEP]' + df['targets3']
  
  if FLAGS.prompt:
    df['text'] = df['text'].apply(lambda x:  'Are they similar?[MASK][SEP]' + x)
  # if FLAGS.use_targets_count:
  #   df['n_targets'] = df.targets_list.apply(len)
  #   df['text'] = df['anchor'] + '[SEP]' + df['target'] + '[SEP]'  + df['context_text'] + '[SEP]' + df['n_targets'].astype(str) + '[SEP]' + df['targets']
  # if FLAGS.add_context:
  # dfc = df.groupby(['anchor'])['context'].apply(list).reset_index(name='contexts')
  # def count_merge(l):
  #   m = defaultdict(int)
  #   for key in l:
  #     m[key] += 1
  #   res = []
  #   for key in m:
  #     res.append(f'{key} {m[key]}')
  #   return ';'.join(res)
  # dfc['contexts'] = dfc.contexts.apply(count_merge)
  # df = df.merge(dfc, on='anchor')
  # df['text'] = df['anchor'] + '[SEP]' + df['target'] + '[SEP]' + df['context'] \
  #               + '[SEP]' + df['context_text'] + '[SEP]' + df['contexts'] + '[SEP]' + df['targets']

  df['num_words'] = df.text.apply(lambda x: len(x.split()))
  if infer:
    df = df.sort_values('num_words', ascending=False)
  if 'score' in df.columns:
    df['cat'] = df.score.apply(lambda x: int(x * 4.))
    qt = QuantileTransformer(n_quantiles=10, random_state=0)
    df['relevance'] = qt.fit_transform(df[['score']])
    df = set_fold(df)
  else:
    gezi.set_worker(df, 80)
  return df


dfs = {}


def get_df(mark='train'):
  if mark in dfs:
    return dfs[mark]
  ifile = f'{FLAGS.root}/train.csv'
  ifile_ = ifile
  if mark == 'test':
    ifile = f'{FLAGS.root}/test.csv'
  df = pd.read_csv(ifile)
  df['sector'] = df.context.apply(lambda x: x[0])
  
  hack_infer = False
  if FLAGS.hack_infer:
    if len(df) < 100:
      df = pd.read_csv(ifile_)
      hack_infer = True
  infer = mark == 'test'
  df = prepare(df, infer=infer)
  if hack_infer:
    df = df[df.fold == FLAGS.fold]
    # df = df.head(1024)

  dfs[mark] = df
  return df


tokenizers = {}


def get_tokenizer(backbone):
  if backbone == 'bradgrimm/patent-cpc-predictor':
    backbone = 'microsoft/deberta-v3-small'
  if 'unilm' in backbone:
    backbone = 'bert-large-cased'

  if backbone in tokenizers:
    return tokenizers[backbone]

  from transformers import AutoTokenizer
  if 'cocolm' in backbone:
    from cocolm.tokenization_cocolm import COCOLMTokenizer as AutoTokenizer
  try:
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_dir)
  except Exception:
    try:
      tokenizer = AutoTokenizer.from_pretrained(backbone)
    except Exception:
      backbone_ = (backbone.split('/')[-1]).replace('_', '-')
      tokenizer = AutoTokenizer.from_pretrained(f'../input/{backbone_}')
  tokenizers[backbone] = tokenizer
  return tokenizer


def encode(text, tokenizer, padding='longest'):
  if 'cocolm' in FLAGS.backbone:
    text = text.replace('[SEP]', '。')
  if not FLAGS.token_types:
    if tokenizer.is_fast:
      res = tokenizer(text,
                      truncation=True,
                      max_length=FLAGS.max_len,
                      padding=padding,
                      return_offsets_mapping=False)
    else:
      res = {}
      input_ids = tokenizer.encode(text)
      res['input_ids'] = gezi.trunct(input_ids, FLAGS.max_len)
      res['attention_mask'] = [1] * len(res['input_ids'])
  else:
    text1, text2 = text.split('[SEP]', 1)
    res = tokenizer(text1,
                    text2,
                    truncation="only_second",
                    max_length=FLAGS.max_len,
                    padding=padding,
                    return_offsets_mapping=False)

  if 'cocolm' in FLAGS.backbone:
    for i in range(len(res['input_ids'])):
      if res['input_ids'][i] == tokenizer.convert_tokens_to_ids('。'):
        res['input_ids'][i] = tokenizer.sep_token_id
  
  if FLAGS.prompt:
    res['prompt_idx'] = res['input_ids'].index(tokenizer.mask_token_id)
  return res


def parse_example(row, tokenizer=None):
  if tokenizer is None:
    tokenizer = get_tokenizer(FLAGS.backbone)
  fe = row.copy()
  padding = 'max_length' if FLAGS.static_inputs_len else 'longest'
  if not FLAGS.two_tower:
    text = row['text']
    fe.update(encode(text, tokenizer, padding))
  else:
    anchor = row['anchor'] + '[SEP]' + row['context']
    res = encode(anchor, tokenizer, padding)
    fe.update({
        'anchor_ids': res['input_ids'],
        'anchor_mask': res['attention_mask'],
    })
    target = row['target'] + '[SEP]' + row['context']
    res = encode(target, tokenizer, padding)
    fe.update({
        'target_ids': res['input_ids'],
        'target_mask': res['attention_mask'],
    })
  if 'score' in row:
    fe['label'] = row['score']
  else:
    fe['label'] = 0.

  if 'targets_list' in fe:
    del fe['targets_list']
  if 'targets_list2' in fe:
    del fe['targets_list2']
  if 'targets_list3' in fe:
    del fe['targets_list3']
  return fe


def get_datasets(valid=True, mark='train'):
  from datasets import Dataset
  df = get_df(mark)
  ic(mark, df)

  ic(FLAGS.backbone)
  tokenizer = get_tokenizer(FLAGS.backbone)
  # ic(tokenizer)

  ds = Dataset.from_pandas(df)

  # num_proc = cpu_count() if FLAGS.pymp else 1
  num_proc = 4 if FLAGS.pymp else 1
  # gezi.try_mkdir(f'{FLAGS.root}/cache')
  records_name = get_records_name()

  ds = ds.map(
      lambda example: parse_example(example, tokenizer=tokenizer),
      remove_columns=ds.column_names,
      batched=False,
      num_proc=num_proc,
      # cache_file_name=f'{FLAGS.root}/cache/{records_name}.infer{int(infer)}.arrow' if not infer else None
  )

  ic(ds)

  ignore_feats = [
      key for key in ds.features if ds.features[key].dtype == 'string' or
      (ds.features[key].dtype == 'list' and
       ds.features[key].feature.dtype == 'string')
  ]
  ic(ignore_feats)

  infer = mark == 'test'
  if infer:
    m = {}
    for key in ignore_feats:
      m[key] = ds[key]
    gezi.set('infer_dict', m)
    ds = ds.remove_columns(ignore_feats)
    return ds

  if not FLAGS.online:
    train_ds = ds.filter(lambda x: x['fold'] != FLAGS.fold, num_proc=num_proc)
  else:
    train_ds = ds
  eval_ds = ds.filter(lambda x: x['fold'] == FLAGS.fold, num_proc=num_proc)

  m = {}
  for key in ignore_feats:
    m[key] = eval_ds[key]
  gezi.set('eval_dict', m)

  # also ok if not remove here
  train_ds = train_ds.remove_columns(ignore_feats)
  eval_ds = eval_ds.remove_columns(ignore_feats)
  ic(train_ds, eval_ds)
  if valid:
    valid_ds = ds.filter(lambda x: x['fold'] == FLAGS.fold, num_proc=num_proc)
    valid_ds = valid_ds.remove_columns(ignore_feats)
    return train_ds, eval_ds, valid_ds
  else:
    return train_ds, eval_ds


def get_dataloaders(valid=True, test_only=False):
  from src.torch.dataset import Dataset
  test_ds = Dataset('test')
  ic(len(test_ds))
  # collate_fn = gezi.DictPadCollate()
  collate_fn = gezi.NpDictPadCollate()
  kwargs = {
      'num_workers': FLAGS.num_workers,
      'pin_memory': FLAGS.pin_memory,
      'persistent_workers': FLAGS.persistent_workers,
      'collate_fn': collate_fn,
  }
  sampler_test = lele.get_sampler(test_ds, shuffle=False)
  test_dl = torch.utils.data.DataLoader(test_ds,
                                        batch_size=gezi.eval_batch_size(),
                                        sampler=sampler_test,
                                        **kwargs)
  if test_only:
    return test_dl

  train_ds = Dataset('train')
  ic(len(train_ds))
  eval_ds = Dataset('valid')
  ic(len(eval_ds))
  ic(len(set(train_ds.df.id) & set(eval_ds.df.id)))
  ic(len(set(train_ds.df.anchor) & set(eval_ds.df.anchor)))
  if valid:
    valid_ds = Dataset('valid')

  # if valid:
  #   train_ds, eval_ds, valid_ds = get_datasets(valid=True)
  # else:
  #   train_ds, eval_ds = get_datasets(valid=False)

  sampler = lele.get_sampler(train_ds, shuffle=True)
  # melt.batch_size 全局总batch大小，FLAGS.batch_size 单个gpu的batch大小，gezi.batch_size做batch的时候考虑兼容distributed情况下的batch_size
  train_dl = torch.utils.data.DataLoader(train_ds,
                                         batch_size=gezi.batch_size(),
                                         sampler=sampler,
                                         drop_last=FLAGS.drop_last,
                                         **kwargs)
  sampler_eval = lele.get_sampler(eval_ds, shuffle=False)
  eval_dl = torch.utils.data.DataLoader(eval_ds,
                                        batch_size=gezi.eval_batch_size(),
                                        sampler=sampler_eval,
                                        **kwargs)
  if valid:
    sampler_valid = lele.get_sampler(valid_ds, shuffle=False)
    valid_dl = torch.utils.data.DataLoader(valid_ds,
                                           batch_size=gezi.eval_batch_size(),
                                           sampler=sampler_valid,
                                           **kwargs)
    return train_dl, eval_dl, valid_dl, test_dl
  else:
    return train_dl, eval_dl, test_dl


def get_tf_datasets(valid=True):
  if valid:
    train_ds, eval_ds, valid_ds = get_datasets(valid=True)
  else:
    train_ds, eval_ds = get_datasets(valid=False)
  collate_fn = gezi.DictPadCollate(return_tensors='tf')
  train_ds = train_ds.to_tf_dataset(
      columns=train_ds.columns,
      label_cols=["label"],
      shuffle=True,
      collate_fn=collate_fn,
      batch_size=gezi.batch_size(),
  )
  eval_ds = eval_ds.to_tf_dataset(
      columns=eval_ds.columns,
      label_cols=["label"],
      shuffle=False,
      collate_fn=collate_fn,
      batch_size=gezi.eval_batch_size(),
  )
  if valid:
    valid_ds = valid_ds.to_tf_dataset(
        columns=valid_ds.columns,
        label_cols=["label"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=gezi.eval_batch_size(),
    )
    return train_ds, eval_ds, valid_ds
  else:
    return train_ds, eval_ds


def create_datasets(valid=True):
  if FLAGS.torch:
    return get_dataloaders(valid)
  else:
    return get_tf_datasets(valid)
