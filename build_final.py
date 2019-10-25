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
from functools import partial
from os import environ
import sys

import pprint

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer

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

def evaluate(dataset_name, estimator):
    dataset = get_dataset(dataset_name)
    estimator = estimator(
        prediction_length=dataset.metadata.prediction_length,
        freq=dataset.metadata.freq,
    )

    print(f"evaluating {estimator} on {dataset}")

    predictor = estimator.train(dataset.train)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset.test, predictor=predictor, num_eval_samples=100
    )

    agg_metrics, item_metrics = Evaluator()(
        ts_it, forecast_it, num_series=len(dataset.test)
    )

    pprint.pprint(agg_metrics)

    eval_dict = agg_metrics
    eval_dict["dataset"] = dataset_name
    eval_dict["estimator"] = type(estimator).__name__
    return eval_dict


if __name__ == "__main__":
    cfg = {
        "learning_rate_decay_factor" : 0.42720667608623236,
        "max_epochs" : 5000,
        "num_batches_per_epoch" : 50,
        "num_cells" : 200,
        "num_layers" : 3,
        "weight_decay" : 3.9297866406356674e-8,
        "dropout_rate" : 0.1,
        "learning_rate" : 0.001,
        "minimum_learning_rate" : 5e-05,
    }
    
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
        logger.info(results)
    except Exception as e:
        logger.warning('Warning on line %d, exception: %s' % (sys.exc_info()[-1].tb_lineno, str(e)))