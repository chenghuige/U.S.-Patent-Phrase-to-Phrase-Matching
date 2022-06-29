#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige
#          \date   2018-10-17 06:52:08.997327
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from email.mime import base

import sys
import os

from absl import flags

FLAGS = flags.FLAGS

from typing import Callable
import pandas as pd

import tensorflow as tf
import torch
from torch import nn
import torch.utils.data
from torch.utils.data import Sampler
#from torch.utils.data import Dataset, ConcatDataset
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
try:
  from apex.normalization import FusedLayerNorm
except ModuleNotFoundError:
  from torch.nn import LayerNorm as FusedLayerNorm
from transformers import PreTrainedModel

import copy
import random
import traceback
import numpy as np
import random
import itertools
from datasets import Dataset

import gezi
from gezi import tqdm
import melt

logging = gezi.logging


def adjust_lrs(x, ratio=None, name='learning_rate_weights'):
  import tensorflow as tf
  if ratio is None:
    ratios = tf.compat.v1.get_collection(name)[-1].numpy()
    # TODO will this hurt performance ? change to use learning rate weights without tf dependence?
    ratios = torch.as_tensor(ratios).cuda()
    x = x * ratios + x.detach() * (1 - ratios)
  else:
    x = x * ratio + x.detach() * (1 - ratio)
  return x


def get_device():
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_weights(model,
                 path,
                 map_location=None,
                 return_checkpoint=False,
                 return_updated=False,
                 renames={},
                 includes=None,
                 excludes=None,
                 to_device=True,
                 eval=True):
  try:
    checkpoint = torch.load(path, map_location=map_location)
  except Exception:
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
  state = checkpoint['state_dict']

  # ic(gezi.get_mem_gb())
  model_ = model.module if hasattr(model, 'module') else model
  full_update = True
  model_state_dict = model_.state_dict()

  def is_ok(key):
    if includes:
      for incl_key in includes:
        if incl_key in key:
          return True
      return False
    if excludes:
      for excl_key in excludes:
        if excl_key in key:
          return False
    return True

  mismatch_ignores = set()
  for key in model_state_dict:
    if state[key].shape != model_state_dict[key].shape:
      mismatch_ignores.add(key)
      full_update = False
  for key in state:
    if key not in model_state_dict:
      full_update = False
  if full_update:
    model_.load_state_dict(state)
  else:
    new_params = model_state_dict
    # ic(new_params.keys())
    if not renames:
      new_params.update({
          k: v
          for k, v in state.items()
          if (k in new_params) and (k not in mismatch_ignores) and is_ok(k)
      })
    else:
      ori = list(renames.keys())[0]
      dest = list(renames.values())[0]
      new_params.update({
          k.replace(ori, dest): v
          for k, v in state.items()
          if k.replace(ori, dest) in new_params and
          k.replace(ori, dest) not in mismatch_ignores
      })

    # ic(new_params.keys())
    model_.load_state_dict(new_params)

  if not return_checkpoint:
    del state
    del checkpoint

  if to_device:
    device = get_device()
    model.to(device)
  if eval:
    model.eval()

  if not return_checkpoint:
    return

  if not return_updated:
    return checkpoint

  updated_params = []
  for name, param in model_.named_parameters():
    if name in state:
      updated_params.append(param)

  return checkpoint, updated_params


# def load_weights(model, path, map_location=None, return_checkpoint=False, return_updated=False, renames={}, to_device=True, eval=True):
#   try:
#     checkpoint = torch.load(path, map_location=map_location)
#   except Exception:
#     checkpoint = torch.load(path, map_location=torch.device('cpu'))
#   state = checkpoint['state_dict']

#   # ic(gezi.get_mem_gb())
#   model_ = model.module if hasattr(model, 'module') else model
#   full_update = True
#   for key in  model_.state_dict():
#     if key not in state:
#       full_update = False
#   if full_update:
#     model_.load_state_dict(state)
#   else:
#     new_params = model_.state_dict()
#     # ic(new_params.keys())
#     if not renames:
#       new_params.update({k:v for k, v in state.items() if k in new_params})
#     else:
#       ori = list(renames.keys())[0]
#       dest = list(renames.values())[0]
#       new_params.update({k.replace(ori, dest): v for k, v in state.items() if k.replace(ori, dest) in new_params})

#     # ic(new_params.keys())
#     model_.load_state_dict(new_params)

#   if not return_checkpoint:
#     del state
#     del checkpoint

#   if to_device:
#     device = get_device()
#     model.to(device)
#   if eval:
#     model.eval()

#   if not return_checkpoint:
#     return

#   if not return_updated:
#     return checkpoint

#   updated_params = []
#   for name, param in model_.named_parameters():
#     if name in state:
#       updated_params.append(param)

#   return checkpoint, updated_params

load = load_weights


def save_model(model, model_dir, model_name='model.pt', fp16=False):
  if fp16:
    model.half()
  state = {
      'state_dict':
          model.state_dict()
          if not hasattr(model, 'module') else model.module.state_dict(),
  }
  if os.path.isdir(model_dir):
    torch.save(state, f'{model_dir}/{model_name}')
  else:
    torch.save(state, model_dir)


def clones(module, N):
  "Produce N identical layers."
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


try:
  import torch
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
  pass
import numpy as np


def torch_(x, cuda=True):
  global device
  if FLAGS.torch_only:
    return x
  for dim in x.shape:
    if dim == 0:
      return x

  # if tf.__version__ < '2':
  x = x.numpy()

  device = gezi.get('device') or device

  if x.dtype == np.int64 or x.dtype == np.int32 or x.dtype == np.float32 or x.dtype == np.float64:
    x = torch.as_tensor(x)
    if cuda:
      x = x.to(device)

  return x


def to_torch(x, y=None, cuda=True, torch_only=False):
  global device
  if torch_only or FLAGS.torch_only:
    if cuda:
      device = gezi.get('device') or device
      for key in x:
        if type(x[key]) != np.ndarray and not isinstance(x[key], (list, tuple)):
          x[key] = x[key].to(device)
      return x, y.to(device)
    else:
      return x, y

  if y is not None:
    y = torch_(y, cuda)

  if not isinstance(x, dict):
    x = torch_(x, cuda)
  else:
    for key in x:
      x[key] = to_torch(x[key], cuda=cuda)

  if y is None:
    return x
  else:
    return x, y


#---------------padding input data

#https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/12


def pad_tensor(vec, pad, val=0, dim=0):
  """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
  pad_size = list(vec.shape)
  pad_size[dim] = pad - vec.size(dim)
  padding = torch.full((1, *pad_size), val, dtype=vec.dtype,
                       device=vec.device).squeeze(0)
  # ic(vec.shape, pad, padding.shape)
  return torch.cat([vec, padding], dim=dim)


class PadCollate2:
  """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

  def __init__(self, dim=0):
    """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
    self.dim = dim

  def pad_collate(self, batch):
    """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
    # find longest sequence
    max_len = max([torch.Tensor(x[0]).shape[self.dim] for x in batch])
    #print('----------', max_len)
    # pad according to max_len
    batch = [(pad_tensor(torch.Tensor(x[0]), pad=max_len, dim=self.dim), x[1])
             for x in batch]
    # stack all
    xs = torch.stack([torch.Tensor(x[0]) for x in batch], dim=0)
    ys = torch.Tensor([x[1] for x in batch])
    return xs, ys

  def __call__(self, batch):
    return self.pad_collate(batch)


class PadCollate:
  """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

  def __init__(self, dim=0):
    """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
    self.dim = dim

  def pad_collate(self, batch):
    """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
    # find longest sequence
    max_len = max([x[0].size(self.dim) for x in batch])
    #print('----------', max_len)
    # pad according to max_len
    batch = [(pad_tensor(x[0], pad=max_len, dim=self.dim), x[1]) for x in batch]
    # stack all
    xs = torch.stack([x[0] for x in batch], dim=0)
    ys = torch.Tensor([x[1] for x in batch])
    return xs, ys

  def __call__(self, batch):
    return self.pad_collate(batch)


# 最常用，输入是numpy array格式
class NpDictPadCollate:
  """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

  def __init__(self, pad_vals={}, dim=0):
    """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
    self.dim = dim
    self.pad_vals = pad_vals

  def pad_collate(self, batch):
    inputs = {}

    if isinstance(batch[0], (list, tuple)):
      ys = [None] * len(batch)
      ys[0] = batch[0][1]
      first_batch = batch[0][0]
      is_tuple = True
    else:
      first_batch = batch[0]
      is_tuple = False
    max_lens = {}

    ignore_keys = set()
    for key, val in first_batch.items():
      if isinstance(val, np.ndarray):
        assert len(val), key
        if type(val[0]) == np.str_:
          ignore_keys.add(key)
          ignore_keys.add(key)
        val = torch.as_tensor(val)
        max_lens[key] = len(val)
      else:
        if isinstance(val, list):
          assert len(val), key
          if type(val[0]) == str:
            ignore_keys.add(key)
            continue
          val = torch.as_tensor(np.asarray(val))
          max_lens[key] = len(val)
      inputs[key] = [val] * len(batch)

    for i in range(1, len(batch)):
      if is_tuple:
        ys[i] = batch[i][1]
        batch_ = batch[i][0]
      else:
        batch_ = batch[i]
      for key, val in batch_.items():
        if key in ignore_keys:
          continue
        if isinstance(val, np.ndarray):
          val = torch.as_tensor(val)
          if len(val) > max_lens[key]:
            max_lens[key] = len(val)
        else:
          if isinstance(val, list):
            try:
              if type(val[0]) == int:
                val = torch.as_tensor(np.asarray(val))
              else:
                val = torch.as_tensor(np.asarray(val)).float()
            except Exception:
              ic(key, val)
            if len(val) > max_lens[key]:
              max_lens[key] = len(val)
        inputs[key][i] = val

    for key, val_list in inputs.items():
      if key in max_lens:
        for i in range(len(val_list)):
          val_list[i] = pad_tensor(val_list[i],
                                   pad=max_lens[key],
                                   val=self.pad_vals.get(key, 0),
                                   dim=self.dim)
          #print(i, val_list[i].shape, max_len[key])

        inputs[key] = torch.stack(val_list, dim=0)
      else:
        # ic(key)
        #... TODO why np.arry.dtype not dp.str_ but <U3 <U4 ?
        inputs[key] = np.asarray(inputs[key])
        try:
          inputs[key] = torch.as_tensor(inputs[key])
        except Exception:
          pass

    if is_tuple:
      ys = torch.as_tensor(np.asarray(ys))
      return inputs, ys
    else:
      return inputs

  def __call__(self, batch):
    return self.pad_collate(batch)


class DictPadCollate:

  def __init__(self, pad_vals={}, dim=0):
    """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
    self.dim = dim
    self.pad_vals = pad_vals

  def pad_collate(self, batch):
    if isinstance(batch[0], (list, tuple)):
      ys = [None] * len(batch)
      ys[0] = batch[0][1]
      first_batch = batch[0][0]
      is_tuple = True
    else:
      first_batch = batch[0]
      is_tuple = False

    inputs = {}
    max_lens = {}

    for key, val in first_batch.items():
      #if not isinstance(val, str):
      if isinstance(val, torch.Tensor):
        if not len(val.size()):
          val = val.expand(1)
        max_lens[key] = val.size(self.dim)
      inputs[key] = [val] * len(batch)

    for i in range(1, len(batch)):
      if is_tuple:
        ys[i] = batch[i][1]
        batch_ = bach[i][0]
      else:
        batch_ = batch[i]
      for key, val in batch_.items():
        #if not isinstance(val, str):
        if isinstance(val, torch.Tensor):
          if not len(val.size()):
            val = val.expand(1)
          if len(val) > max_lens[key]:
            max_lens[key] = val.size(self.dim)
        inputs[key][i] = val

    for key, val_list in inputs.items():
      if key in max_lens:
        for i in range(len(val_list)):
          val_list[i] = pad_tensor(val_list[i],
                                   pad=max_lens[key],
                                   val=self.pad_vals.get(key, 0),
                                   dim=self.dim)
        inputs[key] = torch.stack(val_list, dim=0)
      else:
        inputs[key] = np.array(input[key])

    #list of tensor ->
    if is_tuple:
      ys = torch.stack(ys, dim=0)
      return inputs, ys
    else:
      return inputs

  def __call__(self, batch):
    return self.pad_collate(batch)


# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def keras_init(model, emb=True, linear=True):
  for m in model.modules():
    if emb:
      if isinstance(m, (nn.Embedding, nn.EmbeddingBag)):
        if m.weight.requires_grad:
          nn.init.uniform_(m.weight, -0.05, 0.05)
    if linear:
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def keras_init_children(model, emb=True, linear=False):
  for m in model.children():
    if emb:
      if isinstance(m, (nn.Embedding, nn.EmbeddingBag)):
        if m.weight.requires_grad:
          nn.init.uniform_(m.weight, -0.05, 0.05)
    if linear:
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def keras_weights(x):
  if isinstance(x, (nn.Embedding, nn.EmbeddingBag)):
    if x.weight.requires_grad:
      nn.init.uniform_(x.weight, -0.05, 0.05)
  if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(x.weight)
    nn.init.zeros_(x.bias)


class PytObj(object):

  def __init__(self, x):
    self.x = x

  def numpy(self):
    return self.x


class PytMean(object):

  def __init__(self):
    self._val = 0.
    self.count = 0

    self.is_call = True

  def clear(self):
    self._val = 0
    self.count = 0

  def __call__(self, val=None):
    if val is None:
      return self.result()
    if not self.is_call:
      self.clear()
      self.is_call = True
    self._val += val.item()
    self.count += 1

  def result(self):
    if self.is_call:
      self.is_call = False
    if not self.count:
      val = 0
    else:
      val = self._val / self.count
    # TODO just for compact with tf ..
    return PytObj(val)

  def numpy(self):
    return self.result().numpy()


def predicts(model,
             inputs,
             batch_size=None,
             desc='Predicting',
             dynamic_keys=[],
             mask_key=None):
  with torch.no_grad():
    assert isinstance(inputs, dict)
    assert 0 in inputs
    dataloaders = []
    other_inputs = {}
    for i, inputs_ in inputs.items():
      if isinstance(i, int):
        # ic(i, inputs_.keys())
        inputs__ = {}
        for key in inputs_:
          try:
            if not type(inputs_[key][0]) in [np.str_, str]:
              inputs__[key] = inputs_[key]
          except Exception:
            ic(key)
        inputs_ = inputs__
        dataset = Dataset.from_dict(inputs_)
        device = get_device()
        dataset.set_format(type='torch', device=device)
        assert batch_size, 'need batch size if your inputs is not dataloader but dict'
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        dataloaders.append(dataloader)
      else:
        if not type(inputs_[0]) in [np.str_, str]:
          other_inputs[i] = inputs_

    if other_inputs:
      dataset = Dataset.from_dict(other_inputs)
      device = get_device()
      dataset.set_format(type='torch', device=device)
      assert batch_size, 'need batch size if your inputs is not dataloader but dict'
      dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
      dataloader = iter(dataloader)

    res = None
    total = len(dataloaders[0])
    dataloaders = [iter(x) for x in dataloaders]
    for i in tqdm(range(total), desc=desc):
      inputs_list = [next(dataloaders[j]) for j in range(len(dataloaders))]
      inputs = {}
      for j, inputs_ in enumerate(inputs_list):
        if mask_key is not None:
          max_len = inputs_[mask_key].sum(1).max()
          for key in dynamic_keys + [mask_key]:
            if key in inputs_:
              inputs_[key] = inputs_[key][:, :max_len]
        inputs[j] = inputs_
      if other_inputs:
        inputs.update(next(dataloader))
      preds = model(inputs)
      if isinstance(preds, dict):
        if not res:
          res = {key: [] for key in preds}
        for key in preds:
          res[key].append(gezi.squeeze(preds[key].detach().cpu().numpy()))
      else:
        if not res:
          res = []
        res.append(gezi.squeeze(preds.detach().cpu().numpy()))

    if isinstance(res, dict):
      for key in res:
        try:
          res[key] = np.concatenate(res[key])
        except Exception:
          # l = []
          # for l_ in res[key]:
          #   l.extend(l_)
          # res[key] = l
          res[key] = list(itertools.chain(*res[key]))
    else:
      try:
        res = np.contanate(res)
      except Exception:
        res = list(itertools.chain(*res))

    return res


def predict(model,
            inputs,
            batch_size=None,
            desc='Predicting',
            dynamic_keys=[],
            out_keys=[],
            mask_key=None):
  device = get_device()
  with torch.no_grad():
    input_is_dict = True
    if isinstance(inputs, (list, tuple)):
      if len(inputs) == 2:
        inputs = inputs[0]

    if isinstance(inputs, dict):
      if 0 in inputs:
        return predicts(model, inputs, batch_size, desc, dynamic_keys, mask_key)
      inputs_ = {}
      for key in inputs:
        if (not type(inputs[key][0]) in [np.str_, str]):
          inputs_[key] = inputs[key]
      inputs = inputs_
      # TODO support dynamic length with data collactor padding to max lenght in a batch
      dataset = Dataset.from_dict(inputs)
      dataset.set_format(type='torch', device=device)
      assert batch_size, 'need batch size if your inputs is not dataloader but dict'
      dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    else:
      input_is_dict = False
      dataloader = inputs
    res = None
    pbar = tqdm(dataloader, desc=desc)
    for inputs in pbar:
      if isinstance(inputs, (list, tuple)):
        if len(inputs) == 2:
          inputs = inputs[0]
      if mask_key is not None:
        max_len = inputs[mask_key].sum(1).max()
        for key in dynamic_keys + [mask_key]:
          if key in inputs:
            inputs[key] = inputs[key][:, :max_len]
      if not input_is_dict:
        for key in inputs:
          if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to(device)
      preds = model(inputs)
      if isinstance(preds, dict):
        if not res:
          res = {key: [] for key in preds}
          for key in out_keys:
            res[key] = []
        for key in preds:
          res[key].append(gezi.squeeze(preds[key].detach().cpu().numpy()))
        for key in out_keys:
          if key in inputs:
            if torch.is_tensor(inputs[key]):
              res[key].append(gezi.squeeze(inputs[key].detach().cpu().numpy()))
            else:
              res[key].append(inputs[key])
      else:
        if not res:
          res = []
        res.append(gezi.squeeze(preds.detach().cpu().numpy()))

    if isinstance(res, dict):
      for key in res:
        try:
          res[key] = np.concatenate(res[key])
        except Exception as e:
          # ic(key, e)
          l = []
          for l_ in res[key]:
            l.extend(l_)
          res[key] = l
    else:
      try:
        res = np.contanate(res[key])
      except Exception:
        l = []
        for l_ in res:
          l.extend(l_)
        res = l
    return res


def get_tfrecord_inputs(TFRecordDataset, files, bs=512):
  ds = TFRecordDataset()
  dl = ds.make_batch(bs, filenames=files, return_numpy=True)
  inputs = None
  for x, y in tqdm(dl, total=ds.num_steps, desc=files[0], leave=False):
    if not inputs:
      inputs = {k: list(v) for k, v in x.items()}
      inputs['y'] = list(y)
    else:
      for key in x:
        inputs[key].extend(list(x[key]))
      inputs['y'].extend(list(y))
  for k in inputs:
    inputs[k] = np.asarray(inputs[k])
    try:
      inputs[k] = torch.as_tensor(inputs[k])
    except Exception:
      pass
  return inputs


# https://github.com/ufoym/imbalanced-dataset-sampler/blob/c2ef9d9529f2eb25306aab5e199a99eff455b2cd/torchsampler/imbalanced.py#L40
# from torchsampler import ImbalancedDatasetSampler

# train_loader = torch.utils.data.DataLoader(
#     train_dataset,
#     sampler=ImbalancedDatasetSampler(train_dataset),
#     batch_size=args.batch_size,
#     **kwargs
# )


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
  """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

  def __init__(self,
               dataset,
               indices: list = None,
               num_samples: int = None,
               callback_get_label: Callable = None):
    # if indices is not provided, all elements in the dataset will be considered
    self.indices = list(range(len(dataset))) if indices is None else indices

    # define custom callback
    self.callback_get_label = callback_get_label

    # if num_samples is not provided, draw `len(indices)` samples in each iteration
    self.num_samples = len(self.indices) if num_samples is None else num_samples

    # distribution of classes in the dataset
    df = pd.DataFrame()
    df["label"] = self._get_labels(dataset)
    df.index = self.indices
    df = df.sort_index()

    label_to_count = df["label"].value_counts()

    weights = 1.0 / label_to_count[df["label"]]

    self.weights = torch.DoubleTensor(weights.to_list())

  def _get_labels(self, dataset):
    if self.callback_get_label:
      return self.callback_get_label(dataset)
    # elif isinstance(dataset, torchvision.datasets.MNIST):
    #     return dataset.train_labels.tolist()
    # elif isinstance(dataset, torchvision.datasets.ImageFolder):
    #     return [x[1] for x in dataset.imgs]
    # elif isinstance(dataset, torchvision.datasets.DatasetFolder):
    #     return dataset.samples[:][1]
    # elif isinstance(dataset, torch.utils.data.Subset):
    #     return dataset.dataset.imgs[:][1]
    elif isinstance(dataset, torch.utils.data.Dataset):
      return dataset.get_labels()
    else:
      raise NotImplementedError

  def __iter__(self):
    return (self.indices[i] for i in torch.multinomial(
        self.weights, self.num_samples, replacement=True))

  def __len__(self):
    return self.num_samples


def seed_everything(seed: int):
  random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True


def freeze(model):
  for param in model.parameters():
    param.requires_grad = False


def unfreeze(model):
  for param in model.parameters():
    param.requires_grad = True


def get_word_embeddings(backbone):
  if hasattr(backbone, 'base_model'):
    backbone = backbone.base_model

  if hasattr(backbone, 'word_embedding'):
    # xlnet
    return backbone.word_embedding
  if hasattr(backbone, 'embeddings'):
    # most bert models
    if hasattr(backbone.embeddings, 'word_embedding'):
      return backbone.embeddings.word_embedding
    else:
      # deberta-v2
      return backbone.embeddings.word_embeddings
  if hasattr(backbone, 'shared'):
    # bart
    return backbone.shared
  if hasattr(backbone, 'wte'):
    # gpt2
    return backbone.wte


def get_optimizer_params(model,
                         backbone_lr=None,
                         base_lr=None,
                         weight_decay=False,
                         weight_decay_val=0.01,
                         backbone=None):
  ## 去掉了weight decay 似乎影响不大 不过目前线上的short模型仍然是之前带有weight decay模式训练出来的
  optimizer_parameters = []
  param_optimizer = list(model.named_parameters())
  optimizer_parameters = model.parameters()
  if weight_decay:
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    no_decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay_params = list(map(id, no_decay))
    if backbone_lr is None or base_lr is None:
      # 之前配置 不能完全确定 似乎weight decay降低了集成效果?
      optimizer_parameters = [
          {
              "params": [
                  p for n, p in param_optimizer
                  # if not any(nd in n for nd in no_decay)
                  if not (id(p) in no_decay_params)
              ],
              "weight_decay": weight_decay_val,
          },
          {
              "params": [
                  p for n, p in param_optimizer
                  # if any(nd in n for nd in no_decay)
                  if (id(p) in no_decay_params)
              ],
              "weight_decay": 0.0,
          },
      ]
    else:
      backbone = backbone or model.backbone
      backbone_params = backbone.parameters()
      backbone_params = list(map(id, backbone_params))
      optimizer_parameters = [
          {
              "params": [
                  p for n, p in param_optimizer
                  # if (not any(nd in n for nd in no_decay)) and
                  if (not id(p) in no_decay_params) and
                  (id(p) in backbone_params)
              ],
              "weight_decay": weight_decay_val,
              'lr': backbone_lr,
          },
          {
              "params": [
                  p for n, p in param_optimizer
                  # if (not any(nd in n for nd in no_decay)) and
                  if (not id(p) in no_decay_params) and
                  (not id(p) in backbone_params)
              ],
              "weight_decay": weight_decay_val,
              'lr': base_lr,
          },
          {
              "params": [
                  p for n, p in param_optimizer
                  # any(nd in n for nd in no_decay) and
                  if (id(p) in no_decay_params) and (id(p) in backbone_params)
              ],
              "weight_decay": 0.0,
              'lr': backbone_lr,
          },
          {
              "params": [
                  p for n, p in param_optimizer
                  # if any(nd in n for nd in no_decay) and
                  if (id(p) in no_decay_params) and
                  (not id(p) in backbone_params)
              ],
              "weight_decay": 0.0,
              'lr': base_lr,
          },
      ]
  else:
    if backbone_lr is not None and base_lr is not None:
      backbone = backbone or model.backbone
      backbone_params = backbone.parameters()
      backbone_params = list(map(id, backbone_params))
      # ic([p for p in model.parameters() if (id(p) in backbone_params)])
      # ic([p for p in model.parameters() if (not id(p) in backbone_params)])
      optimizer_parameters = [{
          "params": [
              p for p in model.parameters() if (id(p) in backbone_params)
          ],
          'lr': backbone_lr,
      }, {
          "params": [
              p for p in model.parameters() if (not id(p) in backbone_params)
          ],
          'lr': base_lr,
      }]
  return optimizer_parameters


get_opt_params = get_optimizer_params


class FreezeCallback(object):

  def __init__(self, model, freeze_epochs=1):
    self.model = model
    self.freeze_epochs = freeze_epochs

  def on_train_begin(self):
    if self.freeze_epochs > 0:
      ic('freeze model', self.freeze_epochs)
      freeze(self.model)

  def on_epoch_end(self, epoch):
    if self.freeze_epochs > 0 and (epoch + 1) == self.freeze_epochs:
      ic('unfreeze model', epoch)
      unfreeze(self.model)


class EMA():

  def __init__(self, model, start_epoch=1, decay=0.999):
    self.model = model
    self.start_epoch = start_epoch
    self.decay = decay
    self.shadow = {}
    self.backup = {}
    self.step = 0

  def will_skip(self):
    return melt.epoch() < self.start_epoch

  def on_batch_begin(self, step):
    if self.backup:
      self.restore()

  def on_batch_end(self, step):
    if self.will_skip():
      return False
    if not self.shadow:
      self.register()
    self.update()

  def on_eval_begin(self):
    if self.shadow and self.step > 1:
      self.apply_shadow()

  # def on_eval_end(self):
  #   if self.backup:
  #     self.restore()

  def on_train_end(self):
    if self.shadow:
      self.apply_shadow()

  def register(self):
    for name, param in self.model.named_parameters():
      if param.requires_grad:
        self.shadow[name] = param.data.clone()

  def update(self):
    for name, param in self.model.named_parameters():
      if param.requires_grad:
        assert name in self.shadow
        new_average = (1.0 -
                       self.decay) * param.data + self.decay * self.shadow[name]
        self.shadow[name] = new_average.clone()
    self.step += 1

  def apply_shadow(self):
    for name, param in self.model.named_parameters():
      if param.requires_grad:
        assert name in self.shadow
        self.backup[name] = param.data
        param.data = self.shadow[name]

  def restore(self):
    for name, param in self.model.named_parameters():
      if param.requires_grad:
        assert name in self.backup
        param.data = self.backup[name]
    self.backup = {}


class AWP():

  def __init__(self,
               model,
               start_epoch=1,
               param_name="weight",
               lr=1.,
               eps=0.001):
    self.model = model
    self.param_name = param_name
    self.lr = lr
    self.eps = eps or 0.001
    self.start_epoch = start_epoch
    self.backup = {}
    self.backup_eps = {}

  def will_skip(self, epoch):
    return (self.lr == 0) or (epoch < self.start_epoch)

  def on_retrain_begin(self, epoch=0):
    if self.will_skip(epoch):
      return False

    self.save()
    self.attack()
    return True

  def on_retrain_end(self, epoch=0):
    if self.will_skip(epoch):
      return False

    self.restore()
    return True

  def attack(self):
    e = 1e-6
    for name, param in self.model.named_parameters():
      if param.requires_grad and param.grad is not None and self.param_name in name:
        norm1 = torch.norm(param.grad)
        norm2 = torch.norm(param.data.detach())
        if norm1 != 0 and not torch.isnan(norm1):
          r_at = self.lr * param.grad / (norm1 + e) * (norm2 + e)
          param.data.add_(r_at)
          param.data = torch.min(
              torch.max(param.data, self.backup_eps[name][0]),
              self.backup_eps[name][1])
        # param.data.clamp_(*self.backup_eps[name])

  def save(self):
    for name, param in self.model.named_parameters():
      if param.requires_grad and param.grad is not None and self.param_name in name:
        if name not in self.backup:
          self.backup[name] = param.data.clone()
          grad_eps = self.eps * param.abs().detach()
          self.backup_eps[name] = (
              self.backup[name] - grad_eps,
              self.backup[name] + grad_eps,
          )

  def restore(self,):
    for name, param in self.model.named_parameters():
      if name in self.backup:
        param.data = self.backup[name]
    self.backup = {}
    self.backup_eps = {}


class FGM():

  def __init__(self, model, start_epoch=1, eps=0.25, emb_name='word_embedding'):
    self.model = model
    self.backup = {}
    self.start_epoch = start_epoch
    self.eps = eps or 0.25
    self.emb_name = emb_name

  def will_skip(self, epoch):
    return epoch < self.start_epoch

  def on_retrain_begin(self, epoch=0):
    if self.will_skip(epoch):
      return False

    self.attack()
    return True

  def on_retrain_end(self, epoch=0):
    if self.will_skip(epoch):
      return False

    self.restore()
    return True

  def attack(self):
    # emb_name这个参数要换成你模型中embedding的参数名
    for name, param in self.model.named_parameters():
      if param.requires_grad and self.emb_name in name:
        self.backup[name] = param.data.clone()
        norm = torch.norm(param.grad)
        if norm != 0:
          r_at = self.eps * param.grad / norm
          param.data.add_(r_at)

  def restore(self):
    for name, param in self.model.named_parameters():
      if param.requires_grad and self.emb_name in name:
        assert name in self.backup
        param.data = self.backup[name]
    self.backup = {}


def get_sampler(dataset, shuffle=False):
  if not FLAGS.distributed:
    if shuffle:
      return torch.utils.data.RandomSampler(dataset)
    else:
      return None
  else:
    return torch.utils.data.distributed.DistributedSampler(dataset,
                                                           shuffle=shuffle)


class BucketBatchSampler(Sampler[Iterable[int]]):
  """A batch sampler for sequence bucketing.

    This class creates buckets according to the length of examples. It first sorts the
    lengths and creates index map. Then it groups them into buckets and shuffle
    randomly. This makes each batch has examples of which lengths are almost same. It
    leads the decrement of unnecessary and wasted paddings, hence, you can reduce the
    padded sequence lengths and entire computational costs.

    Args:
        texts: A list of target texts.
        batch_size: The number of examples in each batch.
    """

  def __init__(self,
               lens: List[int],
               batch_size: int,
               drop_last: bool = False,
               shuffle: bool = True,
               max_shift: float = 0.):

    self.lens = lens
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.drop_last = drop_last
    self.max_shift = max_shift

    self.reset()
    # ic(self.buckets.shape[0], self.buckets.shape[0] * batch_size, len(indices))

  def reset(self):
    lens = self.lens
    batch_size = self.batch_size
    if self.max_shift > 0:
      lens = [
          x + np.random.uniform(-self.max_shift, self.max_shift) for x in lens
      ]
    self.lens_ = lens
    indices = np.argsort(lens)

    if len(indices) % batch_size > 0:
      padding = batch_size - len(indices) % batch_size
      indices = np.append(indices, [-1] * padding)

    self.buckets = indices.reshape(-1, batch_size)

    if self.shuffle:
      self.permutation = np.random.permutation(self.buckets.shape[0])
    else:
      self.permutation = np.asarray(range(self.buckets.shape[0]))

  def __len__(self) -> int:
    if not self.drop_last:
      return self.buckets.shape[0]
    else:
      return self.buckets.shape[0] - 1

  def __iter__(self) -> Iterator[Iterable[int]]:
    if not self.drop_last:
      for indices in self.buckets[self.permutation]:
        indices = indices[indices >= 0]
        if self.shuffle:
          np.random.shuffle(indices)
        # print(list(zip(indices[indices >= 0], [self.lens[idx] for idx in indices[indices >= 0]], [self.lens_[idx] for idx in indices[indices >= 0]])))
        yield indices
    else:
      for indices in self.buckets[self.permutation]:
        if indices.sum() == self.batch_size:
          if self.shuffle:
            np.random.shuffle(indices)
          yield indices
    if self.shuffle:
      self.reset()


def get_parameter_groups(module: nn.Module) -> List[Dict[str, Any]]:
  """Get parameter groups for transformer training.

    It is well-known that excluding layer-norm and bias parameters from weight-decay
    leads better performance at training transformer-based models. To achieve that, this
    function creates the separated parameter groups for applying weight-decay and
    ignoring weight-decay.

    Args:
        module: The target module to get the parameters from.

    Returns:
        A list of two parameter groups.
    """
  do_decay = [p for p in module.parameters() if p.ndim < 2]
  no_decay = [p for p in module.parameters() if p.ndim >= 2]
  return [{"params": do_decay}, {"params": no_decay, "weight_decay": 0.0}]


def replace_with_fused_layernorm(module: nn.Module):
  """Replace the normal (PyTorch-vanilla) layer-norms to apex fused layer-norms.

    Args:
        module: The target module to be replaced.
    """
  for submodule in module.modules():
    for name, layer in submodule.named_children():
      if not isinstance(layer, nn.LayerNorm):
        continue

      # Create new fused layer-norm and copy the original parameters.
      new_layer = FusedLayerNorm(layer.normalized_shape, layer.eps)
      new_layer.weight = layer.weight
      new_layer.bias = layer.bias

      # Replace the layer-norm to the new one.
      setattr(submodule, name, new_layer)


def reinit_last_layers(model: PreTrainedModel, num_layers: int):
  """Re-initialize the last-k transformer layers.

    Args:
        model: The target transformer model.
        num_layers: The number of layers to be re-initialized.
    """
  if num_layers > 0:
    base_model = getattr(model, model.base_model_prefix)
    base_model.encoder.layer[-num_layers:].apply(model._init_weights)


def concat_tensors_with_padding(tensor_list: List[torch.Tensor],
                                padding: Union[int, float] = 0) -> torch.Tensor:
  """Concatenate the list of tensors to be a single tensor with paddings.

    Args:
        tensor_list: The list of tensors which have different lengths. They should have
            the shape of `(batch_size, seq_len, dim)` or `(batch_size, seq_len)`.
        padding: The padding value for the tensors. If the tensor is shorter than other
            tensors, than it will be padded with this value. Default is `0`.

    Returns:
        A concatenated single tnesor.
    """
  max_length = max(x.size(1) for x in tensor_list)

  padded_tensor_list = []
  for tensor in tensor_list:
    # This function only supports two and three dimensional tensors.
    if tensor.ndim == 2:
      padding_size = (0, max_length - tensor.size(1))
    elif tensor.ndim == 3:
      padding_size = (0, 0, 0, max_length - tensor.size(1))

    padded_tensor_list.append(F.pad(tensor, padding_size, value=padding))
  return torch.cat(padded_tensor_list)


def compare_models(model_1, model_2, show_val=True):
  models_differ = 0
  for key_item_1, key_item_2 in zip(model_1.state_dict().items(),
                                    model_2.state_dict().items()):
    if torch.equal(key_item_1[1], key_item_2[1]):
      pass
    else:
      models_differ += 1
      if (key_item_1[0] == key_item_2[0]):
        if show_val:
          print('Mismtach found at', key_item_1[0], key_item_1[1],
                key_item_2[1])
        else:
          print('Mismtach found at', key_item_1[0])
      else:
        raise Exception
  if models_differ == 0:
    print('Models match perfectly! :)')


def num_params(model):
  return sum(p.numel() for p in model.parameters())


def num_trainable_params(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def update_scalars(scalars, training=False):
  if not training:
    scalars = gezi.dict_prefix(scalars, 'val_')
  scalars_ = gezi.get('scalars', {})
  scalars_.update(scalars)
  gezi.set('scalars', scalars_)

def set_embedding_parameters_bits(embeddings_path, optim_bits=32):
  """
    https://github.com/huggingface/transformers/issues/14819#issuecomment-1003427930
    """
  import bitsandbytes as bnb
  embedding_types = ("word", "position", "token_type")
  for embedding_type in embedding_types:
    attr_name = f"{embedding_type}_embeddings"

    if hasattr(embeddings_path, attr_name):
      bnb.optim.GlobalOptimManager.get_instance().register_module_override(
          getattr(embeddings_path, attr_name), 'weight',
          {'optim_bits': optim_bits})

def get_embedding(vocab_size, 
                  embedding_dim=None, 
                  embedding_weight=None, 
                  trainable=True, 
                  padding_idx=None):
  logging.info('vocab_size:', vocab_size, 'embedding_weight', embedding_weight)
  embedding = nn.Embedding(vocab_size,
                           embedding_dim,
                           padding_idx=padding_idx)

  if embedding_weight is not None:
    if type(embedding_weight) is str:
      if os.path.exists(embedding_weight):
        embedding_weight = np.load(embedding_weight)
      else:
        embedding_weight = None

    if embedding_weight is not None:   
      if embedding_weight.shape[0] < vocab_size:
        embedding_weight2 = np.random.uniform(-0.05, 0.05,(vocab_size - embedding_weight.shape[0], embedding_dim)) 
        embedding_weight = np.concatenate([embedding_weight, embedding_weight2], axis=0)
      embedding.weight.data.copy_(torch.from_numpy(embedding_weight))
  
  embedding.weight.requires_grad = trainable

  return embedding
