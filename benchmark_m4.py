# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
This example shows how to fit a model and evaluate its predictions.
"""
import pprint
from functools import partial
from sys import exc_info

import pandas as pd

from gluonts.dataset.repository.datasets import get_dataset
#from gluonts.distribution.piecewise_linear import PiecewiseLinearOutput
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer

datasets = [
    "m4_hourly",
    "m4_daily",
    "m4_weekly",
    "m4_monthly",
    "m4_quarterly",
    "m4_yearly",
]

epochs = 1000
num_batches_per_epoch = 50
use_feat_static_cat = True

estimators = [
    partial(
        SimpleFeedForwardEstimator,
        num_hidden_dimensions=[100, 100],
        use_feat_static_cat=use_feat_static_cat,
        cardinality=[6],
        trainer=Trainer(
            epochs=epochs, num_batches_per_epoch=num_batches_per_epoch
        ),
    ),
    partial(
        DeepAREstimator,
        use_feat_static_cat=use_feat_static_cat,
        cardinality=[6],
        trainer=Trainer(
            epochs=epochs, num_batches_per_epoch=num_batches_per_epoch
        ),
    ),
    partial(
        DeepAREstimator,
        num_cells=500,
        num_layers=1,
        use_feat_static_cat=use_feat_static_cat,
        cardinality=[6],
        trainer=Trainer(
            epochs=epochs, num_batches_per_epoch=num_batches_per_epoch
        ),
    ),
#    partial(
#        DeepAREstimator,
#        distr_output=PiecewiseLinearOutput(8),
#        trainer=Trainer(
#            epochs=epochs, num_batches_per_epoch=num_batches_per_epoch
#        ),
#    ),
    partial(
        MQCNNEstimator,
        use_feat_static_cat=use_feat_static_cat,
        cardinality=[6],
        trainer=Trainer(
            epochs=epochs, num_batches_per_epoch=num_batches_per_epoch
        ),
    ),
]


def evaluate(dataset_name, estimator):
    dataset = get_dataset(dataset_name)
    estimator = estimator(
        prediction_length=dataset.metadata.prediction_length,
        freq=dataset.metadata.freq,
    )

    print(f"\nevaluating {estimator} on {dataset_name}")

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

    results = []
    for dataset_name in datasets:
        for estimator in estimators:
            # catch exceptions that are happening during training to avoid failing the whole evaluation
            try:
                results.append(evaluate(dataset_name, estimator))
            except Exception as e:
                print('Warning on line %d, exception:\n%s' % (exc_info()[-1].tb_lineno, str(e)))

    df = pd.DataFrame(results)

    sub_df = df[
        [
            "dataset",
            "estimator",
            "RMSE",
#            "mean_wQuantileLoss",
            "MASE",
            "sMAPE",
            "MSIS",
        ]
    ]

    print(sub_df.to_string())
