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
from datetime import date
from hyperopt import fmin, tpe, hp, space_eval, STATUS_FAIL, STATUS_OK
from hyperopt.mongoexp import MongoTrials
from functools import partial
from os import environ
import sys

import mxnet as mx

from gluonts.dataset.repository.datasets import get_dataset
#from gluonts.time_feature import HourOfDay, DayOfWeek, DayOfMonth, DayOfYear, MonthOfYear
#from gluonts.time_feature.holiday import SpecialDateFeatureSet, CHRISTMAS_DAY, CHRISTMAS_EVE
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer

rand_seed = 42
mx.random.seed(rand_seed, ctx='all')
np.random.seed(rand_seed)
    
########################################################################################################

if ("DATASET" in environ) and ("VERSION" in environ):
    dataset_name = environ.get("DATASET")
    logger.info("Using data set: %s" % dataset_name)
    
    version = environ.get("VERSION")
    logger.info("Using version : %s" % version)
    
    use_cluster = True
else:
    dataset_name = "m4_hourly"
    logger.warning("DATASET not set, using: %s" % dataset_name) 

    version = "test"
    logger.warning("VERSION not set, using: %s" % version)
    
    use_cluster = False

#if dataset_name == "m4_daily":
#    time_features = [DayOfWeek(), DayOfMonth(), DayOfYear(), MonthOfYear()]
#if dataset_name == "m4_hourly":
##    time_features = [HourOfDay(), DayOfWeek(), DayOfMonth(), DayOfYear(), MonthOfYear()]
#    time_features = [HourOfDay(), DayOfWeek()]

num_eval_samples = 1

########################################################################################################
    
def gluon_fcast(cfg):   
    def evaluate(dataset_name, estimator):
        dataset = get_dataset(dataset_name)
        
        estimator = estimator(prediction_length=dataset.metadata.prediction_length, freq=dataset.metadata.freq)
        estimator.ctx = mx.Context("gpu")
        
        logger.info(f"Evaluating {estimator} on {dataset_name}")
        predictor = estimator.train(dataset.train)
        predictor.ctx = mx.Context("gpu")
        
        forecast_it, ts_it = make_evaluation_predictions(dataset.test, predictor=predictor, num_eval_samples=num_eval_samples)
        agg_metrics, item_metrics = Evaluator()(ts_it, forecast_it, num_series=len(dataset.test))
        eval_dict = agg_metrics
        eval_dict["dataset"] = dataset_name
        eval_dict["estimator"] = type(estimator).__name__
        return eval_dict

    ##########################################################################
    
#    if not use_cluster:
#        cfg['max_epochs'] = 2     

    logger.info("Params: %s" % cfg)
    try:    
        estimator = partial(
            DeepAREstimator,
            num_cells=cfg['num_cells'],
            num_layers=cfg['num_layers'],
            dropout_rate=cfg['dropout_rate'],
            use_feat_static_cat=True,
            cardinality=[6],
#            time_features=time_features, 
            trainer=Trainer(
                mx.Context("gpu"),
                epochs=cfg['max_epochs'],
                num_batches_per_epoch=cfg['num_batches_per_epoch'],
                batch_size=cfg['batch_size'],
                
                learning_rate=cfg['learning_rate'],
                learning_rate_decay_factor=cfg['learning_rate_decay_factor'],
                minimum_learning_rate=cfg['minimum_learning_rate'],
                weight_decay=cfg['weight_decay']
            ))
        results = evaluate(dataset_name, estimator)
    except Exception as e:
        logger.warning('Warning on line %d, exception: %s' % (sys.exc_info()[-1].tb_lineno, str(e)))
        return {'loss': None, 'status': STATUS_FAIL, 'cfg' : cfg, 'build_url' : environ.get("BUILD_URL"), 'dataset': dataset_name}

    logger.info(results)
    return {'loss': results['MASE'], 'status': STATUS_OK, 'cfg' : cfg, 'build_url' : environ.get("BUILD_URL"), 'dataset': dataset_name}

# Daily: DeepAREstimator
#	"result" : {
#		"loss" : 3.4650389502919756,
#		"status" : "ok",
#		"cfg" : {
#			"batch_size" : 32,
#			"dropout_rate" : 0.08799143691059126,
#			"learning_rate" : 0.006253177588445521,
#			"learning_rate_decay_factor" : 0.40615719168948966,
#			"max_epochs" : 5000,
#			"minimum_learning_rate" : 0.000008318475256730753,
#			"num_batches_per_epoch" : 60,
#			"num_cells" : 100,
#			"num_layers" : 4,
#			"weight_decay" : 1.7618947010005315e-8
#		},
#		"job_url" : "http://heika:8080/job/hyperopt/16/"
#	},


def call_hyperopt():
    space = {
            'max_epochs'                 : hp.choice('max_epochs', [5000]),
            'num_batches_per_epoch'      : hp.choice('num_batches_per_epoch', [40, 50, 60, 70, 80]),
            'batch_size'                 : hp.choice('batch_size', [32, 64, 128]),

            'num_cells'                  : hp.choice('num_cells', [50, 100, 200, 400]),
            'num_layers'                 : hp.choice('num_layers', [1, 2, 3, 4, 5]),

            'learning_rate'              : hp.uniform('learning_rate', 0.0005, 0.0015),
            'learning_rate_decay_factor' : hp.uniform('learning_rate_decay_factor', 0.1, 0.9),
            'minimum_learning_rate'      : hp.uniform('minimum_learning_rate', 01e-05, 10e-05),
            'weight_decay'               : hp.uniform('weight_decay', 0.5e-08, 5.0e-08),
            'dropout_rate'               : hp.uniform('dropout_rate', 0.05, 0.15),
        }
    
    # Search MongoDB for best trial for exp_key:
    # echo 'db.jobs.find({"exp_key" : "XXX", "result.status" : "ok"}).sort( { "result.loss": 1} ).limit(1).pretty()' | mongo --host heika m4_daily
    # echo 'db.jobs.remove({"exp_key" : "XXX", "result.status" : "new"})' | mongo --host heika

    if use_cluster:
        exp_key = "%s" % str(date.today())
        logger.info("exp_key for this job is: %s" % exp_key)
        trials = MongoTrials('mongo://heika:27017/%s-%s/jobs' % (dataset_name, version), exp_key=exp_key)
        best = fmin(gluon_fcast, space, rstate=np.random.RandomState(rand_seed), algo=tpe.suggest, show_progressbar=False, trials=trials, max_evals=200)
    else:
        best = fmin(gluon_fcast, space, rstate=np.random.RandomState(rand_seed), algo=tpe.suggest, show_progressbar=False, max_evals=5)
        
    params = space_eval(space, best)   
    return(params)
    
if __name__ == "__main__":
    params = call_hyperopt()
    logger.info("Best params: %s" % params)