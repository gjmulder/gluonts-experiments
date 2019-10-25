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

import numpy as np
from datetime import date
from hyperopt import fmin, tpe, hp, space_eval, STATUS_FAIL, STATUS_OK
from hyperopt.mongoexp import MongoTrials
from functools import partial
from os import environ
import sys
#from math import log

from gluonts.dataset.repository.datasets import get_dataset
#from gluonts.distribution.piecewise_linear import PiecewiseLinearOutput
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
#from gluonts.model.seq2seq import MQCNNEstimator
#from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer
    
########################################################################################################
  
rand_seed = 42
import mxnet as mx
mx.random.seed(rand_seed, ctx='all')
np.random.seed(rand_seed)

if "DATASET" in environ:
    dataset_name = environ.get('DATASET')
    logger.info("Using data set: %s" % dataset_name)
    use_cluster = True
else:
    dataset_name = "m4_daily"        
    logger.warning("DATASET not set, using %s" % dataset_name)    
    use_cluster = False
    
def gluon_fcast(cfg):   
    def evaluate(dataset_name, estimator):
        dataset = get_dataset(dataset_name)
        estimator = estimator(
            prediction_length=dataset.metadata.prediction_length,
            freq=dataset.metadata.freq,
        )
    
        logger.info(f"Evaluating {estimator} on {dataset_name}")
    
        predictor = estimator.train(dataset.train)
    
        forecast_it, ts_it = make_evaluation_predictions(
            dataset.test, predictor=predictor, num_eval_samples=100
        )
    
        agg_metrics, item_metrics = Evaluator()(
            ts_it, forecast_it, num_series=len(dataset.test)
        )
    
    #    pprint.pprint(agg_metrics)
    
        eval_dict = agg_metrics
        eval_dict["dataset"] = dataset_name
        eval_dict["estimator"] = type(estimator).__name__
        return eval_dict

    if not use_cluster:
        cfg['max_epochs'] = 2
        
    logger.info("Params: %s" % cfg)
#    use_feat_static_cat = True
#        partial(
#            DeepAREstimator,
#            use_feat_static_cat=use_feat_static_cat,
#            cardinality=[6],
#            trainer=Trainer(
#                epochs=epochs, num_batches_per_epoch=num_batches_per_epoch
#            ),
#        ),
#        partial(
#            DeepAREstimator,
#            num_cells=500,
#            num_layers=1,
#            use_feat_static_cat=use_feat_static_cat,
#            cardinality=[6],
#            trainer=Trainer(
#                epochs=epochs, num_batches_per_epoch=num_batches_per_epoch
#            ),
#        ),
#        partial(
#            DeepAREstimator,
#            distr_output=PiecewiseLinearOutput(8),
#            trainer=Trainer(
#                epochs=epochs, num_batches_per_epoch=num_batches_per_epoch
#            ),
#        ),
#        partial(
#            MQCNNEstimator,
#            use_feat_static_cat=use_feat_static_cat,
#            cardinality=[6],
#            trainer=Trainer(
#                epochs=epochs, num_batches_per_epoch=num_batches_per_epoch
#            ),
#        ),
    
    
    ########################################################################################################
    # catch exceptions that are happening during training to avoid failing the whole evaluation
    try:    
        estimator = partial(
            DeepAREstimator,
            num_cells=cfg['num_cells'],
            num_layers=cfg['num_layers'],
            dropout_rate=cfg['dropout_rate'],
            use_feat_static_cat=True,
            cardinality=[6],
            trainer=Trainer(
                epochs=cfg['max_epochs'],
                num_batches_per_epoch=cfg['num_batches_per_epoch'],
                learning_rate=cfg['learning_rate'],
                learning_rate_decay_factor=cfg['learning_rate_decay_factor'],
                minimum_learning_rate=cfg['minimum_learning_rate'],
                weight_decay=cfg['weight_decay']
            ),
        )
        results = evaluate(dataset_name, estimator)
    except Exception as e:
        logger.warning('Warning on line %d, exception: %s' % (sys.exc_info()[-1].tb_lineno, str(e)))
        return {'loss': None, 'status': STATUS_FAIL, 'cfg' : cfg}

    logger.info(results)
    return {'loss': results['MASE'], 'status': STATUS_OK, 'cfg' : cfg}

# Daily: {'learning_rate_decay_factor': 0.575370116706172, 'max_epochs': 1000, 'num_batches_per_epoch': 100,
#         'num_cells': 200, 'num_layers': 2, 'weight_decay': 2.432055488706233e-08}
    
# Daily: {'learning_rate_decay_factor': 0.5006305686152246, 'max_epochs': 1000, 'num_batches_per_epoch': 100,
#         'num_cells': 100, 'num_layers': 2, 'weight_decay': 3.946049025967661e-08}

# Daily:
#		"loss" : 3.2569833100061882,
#		"status" : "ok",
#		"cfg" : {
#			"learning_rate_decay_factor" : 0.42720667608623236,
#			"max_epochs" : 5000,
#			"num_batches_per_epoch" : 50,
#			"num_cells" : 200,
#			"num_layers" : 3,
#			"weight_decay" : 3.9297866406356674e-8
#		}

def call_hyperopt():
    space = {
            'max_epochs'                 : hp.choice('max_epochs', [5000]),
            'num_batches_per_epoch'      : hp.choice('num_batches_per_epoch', [30, 40, 50, 60]),

            'num_cells'                  : hp.choice('num_cells', [50, 100, 200, 400]),
            'num_layers'                 : hp.choice('num_layers', [2, 3, 4]),

            'learning_rate'              : hp.uniform('learning_rate', 0.010, 0.005),
            'learning_rate_decay_factor' : hp.uniform('learning_rate_decay_factor', 0.3, 0.5),
            'minimum_learning_rate'      : hp.uniform('minimum_learning_rate', 1e-05, 10e-05),
            'weight_decay'               : hp.loguniform('weight_decay', -17.5, -16.7),
            'dropout_rate'               : hp.uniform('dropout_rate', 0.05, 0.15),
        }
    
    # Search MongoDB for best trial for exp_key:
    # echo 'db.jobs.find({"exp_key" : "XXX", "result.status" : "ok"}).sort( { "result.loss": 1} ).limit(1).pretty()' | mongo --host heika m4_daily
    # echo 'db.jobs.remove({"exp_key" : "XXX", "result.status" : "new"})' | mongo --host heika

    if use_cluster:
        exp_key = "%s_%s" % (str(date.today()), version)
        logger.info("exp_key for this job is: %s" % exp_key)
        trials = MongoTrials('mongo://heika:27017/%s/jobs' % dataset_name, exp_key=exp_key)
        best = fmin(gluon_fcast, space, rstate=np.random.RandomState(rand_seed), algo=tpe.suggest, trials=trials, max_evals=200, show_progressbar=False)
    else:
        best = fmin(gluon_fcast, space, rstate=np.random.RandomState(rand_seed), algo=tpe.suggest, max_evals=5, show_progressbar=False)
        
    params = space_eval(space, best)   
    return(params)
    
if __name__ == "__main__":
    params = call_hyperopt()
    logger.info("Best params: %s" % params)