#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:09:36 2019

@author: mulderg
"""

version = "0.1"

from logging import basicConfig, getLogger
#from logging import DEBUG as log_level
from logging import INFO as log_level
basicConfig(level = log_level,
            format  = '%(asctime)s %(levelname)-8s %(module)-20s: %(message)s',
            datefmt ='%Y-%m-%d %H:%M:%S')
logger = getLogger(__name__)

import mxnet as mx
import numpy as np

rand_seed = 42
mx.random.seed(rand_seed, ctx='all')
np.random.seed(rand_seed)

#datasets = [
#    "m4_hourly",
#    "m4_daily",
#    "m4_weekly",
#    "m4_monthly",
#    "m4_quarterly",
#    "m4_yearly",
#]

use_cluster = False
dataset_name = "m4_daily"
max_evals = 100