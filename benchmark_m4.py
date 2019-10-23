#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:09:36 2019

@author: mulderg
"""

import config
logger = config.logger

from functools import partial
import sys
from datetime import date
from numpy import random
#import pandas as pd

########################################################################################################

from gluonts.dataset.repository.datasets import get_dataset
#from gluonts.distribution.piecewise_linear import PiecewiseLinearOutput
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
#from gluonts.model.seq2seq import MQCNNEstimator
#from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer

from hyperopt import fmin, tpe, hp, space_eval, STATUS_FAIL, STATUS_OK
from hyperopt.mongoexp import MongoTrials

########################################################################################################

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
    
def gluon_fcast(cfg):
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
    
    estimator = partial(
        DeepAREstimator,
        num_cells=cfg['num_cells'],
        num_layers=cfg['num_layers'],
        use_feat_static_cat=True,
        cardinality=[6],
        trainer=Trainer(
            epochs=cfg['max_epochs'],
            num_batches_per_epoch=cfg['num_batches_per_epoch'],
            learning_rate_decay_factor=cfg['learning_rate_decay_factor'],
            weight_decay=cfg['weight_decay']
        ),
    )
    # catch exceptions that are happening during training to avoid failing the whole evaluation
    try:
        results = evaluate(config.dataset_name, estimator)
    except Exception as e:
        logger.warn('Warning on line %d, exception: %s' % (sys.exc_info()[-1].tb_lineno, str(e)))
        return {'loss': None, 'status': STATUS_FAIL}

    logger.info(results)
    return {'loss': results['MASE'], 'status': STATUS_OK}
    
def call_hyperopt():
    space = {
            'num_cells'                  : hp.choice('num_cells', [50, 100, 200]),
            'num_layers'                 : hp.choice('num_layers', [1, 2, 3]),

            'max_epochs'                 : hp.choice('max_epochs', [1000]),
            'num_batches_per_epoch'      : hp.choice('num_batches_per_epoch', [25, 50, 100]),
            'learning_rate_decay_factor' : hp.uniform('learning_rate_decay_factor', 0.5, 0.9),
            'weight_decay'               : hp.loguniform('weight_decay', -19, -17),
        }
    
    # Search MongoDB for best trial for exp_key:
    # echo 'db.jobs.find({"exp_key" : "XXXX-YYYY-MM-DD", "result.status" : "ok"}).sort( { "result.loss": 1} ).limit(1).pretty()' | mongo hyperopt_db
    if config.use_cluster:
        exp_key = "%s_%s" % (str(date.today()), config.version)
        logger.info("exp_key for this job is: %s" % exp_key)
        trials = MongoTrials('mongo://heika:27017/%s/jobs' % config.dataset_name, exp_key=exp_key)
        best = fmin(gluon_fcast, space, rstate=random.RandomState(config.rand_seed), algo=tpe.suggest, trials=trials, max_evals=config.max_evals, show_progressbar=False)
    else:
        best = fmin(gluon_fcast, space, rstate=random.RandomState(config.rand_seed), algo=tpe.suggest, max_evals=20, show_progressbar=False)
        
    params = space_eval(space, best)   
    return(params)
    
if __name__ == "__main__":
    params = call_hyperopt()
    logger.info("Best params: %s" % params)