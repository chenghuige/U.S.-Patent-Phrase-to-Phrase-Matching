#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   common.py
#        \author   chenghuige  
#          \date   2022-04-19 11:42:25.883552
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os
import glob

from absl import app, flags
FLAGS = flags.FLAGS

import numpy as np
import scipy
import sklearn
import pandas as pd
from functools import partial
from collections import Counter, OrderedDict, defaultdict
import json
import re
from pathlib import Path
import itertools
import dill

try:
  import cudf
except Exception:
  pass

from gezi import tqdm, rtqdm, logging
tqdm.pandas()
logger = logging.logger
logger2 = logging.logger2

from multiprocessing import Pool, Manager, cpu_count
from joblib import Parallel, delayed
try:
  import pymp 
except Exception:
  pass

import tensorflow as tf
import torch

try:
  from rich_dataframe import prettify
except Exception:
  pass
from IPython.display import display_html, display
import plotly.express as px

import melt as mt
import lele
import husky 
import gezi

PERCENTILES = [.25,.5,.75,.9,.95,.99]
PERCENTILES2 = [.25,.5,.75,.9,.95,.99,.999]
