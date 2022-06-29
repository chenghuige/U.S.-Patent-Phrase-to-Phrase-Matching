#!/usr/bin/env python
# coding: utf-8

from gezi.common import *
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
sys.path.append('..')
os.environ["WANDB_SILENT"] = "true"

from src.eval import *
from src.util import *
from src.config import *
from src.ensemble_conf import *
from src.postprocess import *
from src import config

pd.set_option('display.float_format', lambda x: '%.02f' % x)

# ------------------ config
             
def main(_): 
  from src import config
  config.init()
  # FLAGS.eval_ori = False
  from absl import flags
  FLAGS = flags.FLAGS
  FLAGS.show_keys = ['score']
  show_keys_ = ['score']
  show_keys = ['folds', 'fold']
  show_keys += FLAGS.show_keys
  metric = show_keys[-1]
  if (len(sys.argv) > 1 and sys.argv[1].startswith('on')) or FLAGS.online:
    mark = 'online'
  else:
    mark = 'offline'

  if FLAGS.ensemble_metrics:
    mns1, mns2, mns3, mns = [], [], [FLAGS.mn], [FLAGS.mn]
    v1 = v2 = v3 = v = RUN_VERSION
  else:
    from src.ensemble_conf import mns1, mns2, mns3, mns, v1, v2, v3, v
    
  folds = FLAGS.folds if mark != 'online' else 1
  scores = []
  x_ = None
  root = f'../working/{mark}/{v}'
  xs_ = []
  df_gt = pd.read_csv(f'{FLAGS.root}/train.csv')
  has_missings = False
  has_bads = False
  model_dirs_ = None
  for fold in range(folds):
    model_dirs = [f'../working/{mark}/{v1}/{fold}/{mn}' for mn in mns1]
    model_dirs += [f'../working/{mark}/{v2}/{fold}/{mn}' for mn in mns2]
    model_dirs += [f'../working/{mark}/{v3}/{fold}/{mn}' for mn in mns3]
    if fold == 0:
      ic(model_dirs)
      # gezi.restore_configs(model_dirs[0])
    missings = [
        model_dir for model_dir in model_dirs
        if not os.path.exists(f'{model_dir}/valid.pkl')
    ]
    missings2 = [
        model_dir for model_dir in model_dirs
        if not os.path.exists(f'{model_dir}/log.txt')
    ]
    if missings:
      ic(fold, missings)
      has_missings = True
    try: 
      bads = [
        model_dir for model_dir in model_dirs
        if os.path.exists(f'{model_dir}/metrics.csv') and pd.read_csv(f'{model_dir}/metrics.csv')[FLAGS.show_keys[0]].values[-1] < 0.4
      ]
      if bads:
        ic(fold, bads)
        has_bads = True
    except Exception:
      pass
    if fold == 0:
      model_dirs_ = model_dirs.copy()

  model_dir = model_dirs[0]
  config = FLAGS.flag_values_dict().copy()
  config['models'] = mns
  config['model_weights'] = list(zip(mns, weights[:len(mns)]))
  mns_name = str(len(mns)) + '_' + '|'.join(mns)
  
  ic(has_missings, has_bads)
  # if has_missings or has_bads:
  #   exit(0)
  # if has_bads:
  #   exit(0)
  import copy
  # TODO not work like FLAGS.kmethod.. still change
  FLAGS_ = copy.deepcopy(FLAGS)
  kmethod = FLAGS.kmethod

  for fold in range(folds):
    model_dirs = [f'../working/{mark}/{v1}/{fold}/{mn}' for mn in mns1]
    model_dirs += [f'../working/{mark}/{v2}/{fold}/{mn}' for mn in mns2]
    model_dirs += [f'../working/{mark}/{v3}/{fold}/{mn}' for mn in mns3]
    
    if len(mns) == 1:
      d = pd.read_csv(f'{model_dirs[0]}/metrics.csv')
      metrics = d.to_dict('records')[-1]
      metrics = {k: v for k, v in metrics.items() if k in FLAGS.show_keys}
      for mn in mns:
        gezi.pprint_dict(gezi.merge_dicts({'fold': fold, 'mn': mn}, metrics))
    
    xs = [gezi.load(f'{model_dir}/valid.pkl') for model_dir in model_dirs]

    model_dir = model_dirs[0]
    l = []
    ensembler = gezi.Ensembler()
    for i, x in tqdm(enumerate(xs), total=len(xs), desc='convert', leave=False, ascii=True):
    # for i, x in enumerate(xs):
      x = {k: v  for k, v in x.items() if k in ['pred', 'id', 'label', 'labels']}
      try:
        d = pd.read_csv(f'{model_dirs[i]}/metrics.csv')
        metrics = d.to_dict('records')[-1]
        metrics = {k: v for k, v in metrics.items() if k in FLAGS.show_keys}
        l.append(gezi.merge_dicts({ 'fold': fold, 'mn': mns[i]}, metrics))
        if (d[FLAGS.show_keys[0]].values[-1] < 0.5):
          ic(model_dirs[i], d[metric].values[-1])
      except Exception:
        pass
      gezi.restore_configs(model_dirs[i])
      FLAGS.show_keys = show_keys_
      
      # bs = 128
      # ofile = '../working/x.pkl'
      # command = f'python ./infer.py {model_dirs[i]} {bs} {ofile} 0 1'
      # ic(i, model_dirs[i], bs, command)
      # os.system(command)
      # x = gezi.load(ofile)
  
      x['pred'] = to_pred(x['pred'])
      ic(i, weights[i])
      ensembler.add(x, weights[i])
    if l:
      gezi.pprint_df(pd.DataFrame(l))

    x = ensembler.finalize()
    xs_.append(x.copy())

    x['pred'] = to_pred(x['pred'])
    if not 'label' in x:
      x['label'] = x['labels']
    res = {'score': calc_metric(x['label'], x['pred'])}
    res['fold'] = fold
    
    scores.append(res.copy())
    if len(scores) > 1:
      df = pd.DataFrame(scores)
      answer = dict(df.mean())
      answer['fold'] = len(scores)
      gezi.pprint_df(df, keys=show_keys)
      gezi.pprint_dict(answer, keys=show_keys)
    else:
      res['fold'] = 1
      gezi.pprint_dict(res, keys=show_keys)
      
  num_folds = len(scores)
  if num_folds > 1:
    x = gezi.merge_array_dicts(xs_)
    if not 'label' in x:
      x['label'] = x['labels']
    x['pred'] = to_pred(x['pred'])
    res = {'score': calc_metric(x['label'], x['pred'])}

    res['fold'] = num_folds
    ic(res)
  
  if mark == 'offline' and num_folds == folds:
    res['mns'] = mns_name
    writer = gezi.MetricsWriter(f'../working/{mark}/{v}/ensemble.csv')
    writer.write(res)
    d = pd.read_csv(writer.metric_file)[[metric, 'mns']].drop_duplicates()
    print('by time:')
    gezi.pprint_df(d[[metric, 'mns']].tail(5))
    d = d.sort_values(metric, ascending=False)
    print('by score:')
    gezi.pprint_df(d[[metric, 'mns']].head(5))
    pre_best = d[metric].values[0]
  
    ic(res[metric] >= pre_best)
    gezi.pprint_df(df, keys=show_keys)
    gezi.pprint_dict(res, keys=show_keys)
    ic(num_folds, mns, len(mns), pre_best, res[metric])
    x['id'] = df_gt['id'].values
    df_pred = to_df(x)
    # pred_name = FLAGS.save_pred_name or 'valid'
    pred_name = 'valid'
    ic(f'{root}/{pred_name}.csv')
    # # if SAVE_PRED or (res[metric] >= pre_best) or FLAGS.save_pred or FLAGS.save_pred_name:
    # if SAVE_PRED or (res[metric] >= pre_best):
    #   df.to_csv(f'{root}/{pred_name}.csv', index=False)
    #   df_pred.reset_index().to_feather(f'{root}/{pred_name}.fea')
    #   gezi.save(x, f'{root}/{pred_name}.pkl')
    # elif FLAGS.save_csv:
    #   df_pred.to_csv(f'{root}/{pred_name}.csv', index=False)
    #   df_pred.reset_index().to_feather(f'{root}/{pred_name}.fea')
  
  # TODO upload multi folds models of offline?
  FLAGS = FLAGS_
  FLAGS.kmethod = kmethod
  if FLAGS.kaggle_idx is not None:
    ic(model_dirs_)
    if mark == 'online':
      for i, model_dir in enumerate(model_dirs_):
        kaggle_idx = FLAGS.kaggle_idx + i
        ic(i, kaggle_idx, model_dir)
        gezi.prepare_kaggle_dataset('usp-model', model_dir=model_dir, kaggle_idx=kaggle_idx, fold=0, method=FLAGS.kmethod, exit_last=False)
    else:
      # notice you version should not use 0.. other wise offline/0/0/ fail here, start from 1 
      fold_mark = '/0/'
      idx = 0
      if FLAGS.kaggle_online:
        folds += 1
      for i, model_dir_ in enumerate(model_dirs_):
        for j in range(FLAGS.kaggle_folds):
          kaggle_idx = FLAGS.kaggle_idx + idx
          fold = (FLAGS.kaggle_fold + idx) % folds
          model_dir = model_dir_.replace(fold_mark, f'/{fold}/')
          ic(i, j, fold, kaggle_idx, model_dir, FLAGS.kmethod)
          gezi.prepare_kaggle_dataset('usp-model', model_dir=model_dir, kaggle_idx=kaggle_idx, fold=0, method=FLAGS.kmethod, exit_last=False)
          idx += 1
        if FLAGS.kaggle_online:
          kaggle_idx = FLAGS.kaggle_idx + idx
          model_dir = model_dir_.replace('/offline/', '/online/')
          ic(i, kaggle_idx, model_dir, FLAGS.kmethod)
          gezi.prepare_kaggle_dataset('usp-model', model_dir=model_dir, kaggle_idx=kaggle_idx, fold=0,  method=FLAGS.kmethod, exit_last=False)
          idx += 1
    # gezi.system('kaggle-update')
          
if __name__ == '__main__':
  app.run(main)  