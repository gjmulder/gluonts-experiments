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

import pandas as pd
import numpy as np

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset, TrainDatasets, MetaData

#from gluonts.distribution.piecewise_linear import PiecewiseLinearOutput
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
#from gluonts.model.seq2seq import MQCNNEstimator
#from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer

epochs = 2000
num_batches_per_epoch = 100

datasets = [
    "m4_hourly",
    #    "m4_daily",
    #    "m4_weekly",
    #    "m4_monthly",
    #    "m4_quarterly",
    #    "m4_yearly",
]


def generate_data(num_series, num_steps, period=24, mu=1, sigma=0.3):
    # create target: noise + pattern
    # noise
    noise = np.random.normal(mu, sigma, size=(num_series, num_steps))

    # pattern - sinusoid with different phase
    sin_minumPi_Pi = np.sin(
        np.tile(np.linspace(-np.pi, np.pi, period), int(num_steps / period)))
    sin_Zero_2Pi = np.sin(
        np.tile(np.linspace(0, 2 * np.pi, 24), int(num_steps / period)))

    pattern = np.concatenate((np.tile(sin_minumPi_Pi.reshape(1, -1),
                                      (int(np.ceil(num_series / 2)), 1)),
                              np.tile(sin_Zero_2Pi.reshape(1, -1),
                                      (int(np.floor(num_series / 2)), 1))
                              ),
                             axis=0
                             )

    target = noise + pattern

    # create time features: use target one period earlier, append with zeros
    feat_dynamic_real = np.concatenate((np.zeros((num_series, period)),
                                        target[:, :-period]
                                        ),
                                       axis=1
                                       )

    # create categorical static feats: use the sinusoid type as a categorical feature
    feat_static_cat = np.concatenate((np.zeros(int(np.ceil(num_series / 2))),
                                      np.ones(int(np.floor(num_series / 2)))
                                      ),
                                     axis=0
                                     )

    return target, feat_dynamic_real, feat_static_cat


def get_datasets(dataset_name):
    # define the parameters of the dataset
    meta = {
        'num_series': 10,
        'num_steps': 24 * 7,
        'prediction_length': 24,
        'freq': '1H',
        'start': [pd.Timestamp("01-01-2019", freq='1H') for _ in range(100)]
    }

    target, feat_dynamic_real, feat_static_cat = generate_data(meta['num_series'],
                                                               meta['num_steps'],
                                                               meta['prediction_length']
                                                               )

    train_ds = ListDataset([{FieldName.TARGET: target,
                             FieldName.START: start,
                             FieldName.FEAT_DYNAMIC_REAL: fdr,
                             FieldName.FEAT_STATIC_CAT: fsc}
                            for (target, start, fdr, fsc) in zip(target[:, :-meta['prediction_length']],
                                                                 meta['start'],
                                                                 feat_dynamic_real[:, :-
                                                                                   meta['prediction_length']],
                                                                 feat_static_cat)],
                           freq=meta['freq'])

    test_ds = ListDataset([{FieldName.TARGET: target,
                            FieldName.START: start,
                            FieldName.FEAT_DYNAMIC_REAL: fdr,
                            FieldName.FEAT_STATIC_CAT: fsc}
                           for (target, start, fdr, fsc) in zip(target,
                                                                meta['start'],
                                                                feat_dynamic_real,
                                                                feat_static_cat)],
                          freq=meta['freq'])

    return TrainDatasets(metadata=meta, train=train_ds, test=test_ds)


estimators = [
    #    partial(
    #        SimpleFeedForwardEstimator,
    #        trainer=Trainer(
    #            epochs=epochs, num_batches_per_epoch=num_batches_per_epoch
    #        ),
    #    ),
    partial(
        DeepAREstimator,
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
    #    partial(
    #        MQCNNEstimator,
    #        trainer=Trainer(
    #            epochs=epochs, num_batches_per_epoch=num_batches_per_epoch
    #        ),
    #    ),
]


def evaluate(dataset_name, estimator):
    dataset = get_datasets(dataset_name)
    estimator = estimator(
        prediction_length=dataset.metadata['prediction_length'],
        freq=dataset.metadata['freq']
    )

    print(f"evaluating {estimator} on {dataset_name}")

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
            #            # catch exceptions that are happening during training to avoid failing the whole evaluation
            #            try:
            results.append(evaluate(dataset_name, estimator))
#            except Exception as e:
#                print(str(e))

    df = pd.DataFrame(results)

#    sub_df = df[
#        [
#            "dataset",
#            "estimator",
# "RMSE",
# "mean_wQuantileLoss",
#            "MASE",
#            "sMAPE",
# "MSIS",
#        ]
#    ]
#
#    print(sub_df.to_string())
    print(df.to_string())
