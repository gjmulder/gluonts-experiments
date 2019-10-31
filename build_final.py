#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:09:36 2019

@author: mulderg
"""

from logging import basicConfig, getLogger
#from logging import DEBUG as log_level
from logging import INFO as log_level
basicConfig(level = log_level,
            format  = '%(asctime)s %(levelname)-8s %(module)-20s: %(message)s',
            datefmt ='%Y-%m-%d %H:%M:%S')
logger = getLogger(__name__)

import numpy as np
from functools import partial
from os import environ

import mxnet as mx

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.time_feature import HourOfDay, DayOfWeek, DayOfMonth, DayOfYear, MonthOfYear
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.trainer import Trainer

rand_seed = 42
mx.random.seed(rand_seed, ctx='all')
np.random.seed(rand_seed)

########################################################################################################

if "DATASET" in environ:
    dataset_name = environ.get('DATASET')
    logger.info("Using data set: %s" % dataset_name)
    use_cluster = True
else:
    dataset_name = "m4_hourly"        
    logger.warning("DATASET not set, using %s" % dataset_name)    
    use_cluster = False

#if dataset_name == "m4_daily":
#    time_features = [DayOfWeek(), DayOfMonth(), DayOfYear(), MonthOfYear()]
#if dataset_name == "m4_hourly":
##    time_features = [HourOfDay(), DayOfWeek(), DayOfMonth(), DayOfYear(), MonthOfYear()]
#    time_features = [HourOfDay(), DayOfWeek()]

num_eval_samples = 100

########################################################################################################

def evaluate(dataset_name, estimator):
    dataset = get_dataset(dataset_name)
    estimator = estimator(prediction_length=dataset.metadata.prediction_length, freq=dataset.metadata.freq)
    estimator.ctx = mx.Context("cpu")

    logger.info(f"evaluating {estimator} on {dataset}")

    predictor = estimator.train(dataset.train)
    predictor.ctx = mx.Context("cpu")

    forecast_it, ts_it = make_evaluation_predictions(dataset.test, predictor=predictor, num_eval_samples=num_eval_samples)
    agg_metrics, item_metrics = Evaluator()(ts_it, forecast_it, num_series=len(dataset.test))
    eval_dict = agg_metrics
    eval_dict["dataset"] = dataset_name
    eval_dict["estimator"] = type(estimator).__name__
    return eval_dict

if __name__ == "__main__":
    cfg = {
        "max_epochs" : 20,
        "num_batches_per_epoch" : 60,
        "num_cells" : 100,
        "num_layers" : 4,
        "add_trend" : False
    }
    
    estimator = partial(
        DeepStateEstimator,
        num_cells=cfg['num_cells'],
        num_layers=cfg['num_layers'],
        use_feat_static_cat=True,
        cardinality=[6],
        add_trend=cfg['add_trend'],
#        time_features=time_features, 
        num_eval_samples=num_eval_samples,
        trainer=Trainer(
            mx.Context("cpu"),
            epochs=cfg['max_epochs'],
            num_batches_per_epoch=cfg['num_batches_per_epoch'])
        )

    results = evaluate(dataset_name, estimator)
    logger.info(results)