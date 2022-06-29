import husky.callbacks
from husky.train import train
import husky.optimization
from husky.callbacks.tqdm_progress_bar import TQDMProgressBar
from husky.callbacks.tqdm_progress_bar import TQDMProgressBar as ProgressBar
from husky.ema import ExponentialMovingAverage 
import tensorflow as tf

# TODO 多gpu下似乎不对 failed to convert a NumPy array to a Tensor, 单gpu kaggle验证ok
def predict(model, inputs, batch_size=None, desc='Predicting'):
  import numpy as np
  import tensorflow as tf
  import gezi
  from gezi import tqdm
  import melt as mt
  strategy = mt.distributed.get_strategy()
  with strategy.scope():
    if isinstance(inputs, dict):
      dataloader = tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size)
    else:
      dataloader = inputs
    res = None
    # dataloader = strategy.experimental_distribute_dataset(dataloader)
    for inputs in tqdm(dataloader, desc=desc):
      preds = model.predict_on_batch(inputs)
      if isinstance(preds, dict):
        if not res:
          res = {key: [] for key in preds}
        for key in preds:
          res[key].append(gezi.squeeze(preds[key]))
      else:
        if not res:
          res = []
        res.append(gezi.squeeze(preds))
    
    if isinstance(res, dict):
      for key in res:
        res[key] = np.concatenate(res[key])
    else:
      res = np.contanate(res[key])
    return res
