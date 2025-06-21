#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Script for evaluating multiple AutoML frameworks on regression tasks
using the RCA dataset. Includes AutoGluon, AutoKeras, H2O, FLAML, and TPOT.

Author: Adapted version
"""

import os
import time
import json
import glob as gl
import warnings
import numpy as np
import pandas as pd
import pylab as pl

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from autosklearn.regression import AutoSklearnRegressor
from flaml import AutoML
from tpot import TPOTRegressor
from autokeras import StructuredDataRegressor
from autogluon.tabular import TabularDataset, TabularPredictor

import h2o
from h2o.automl import H2OAutoML
from read_data_rca import *

warnings.filterwarnings('ignore')
h2o.init()

# Fix deprecated NumPy types
np.float = np.float64
np.bool = np.bool_
np.object = object

# Global settings
pd.options.display.float_format = '{:.3f}'.format
basename = 'rca_automl_'
time_budget = 120
n_runs = 30
n_splits = 5
epochs = 100

# Utility function for permutation feature importance
def permutation_feature_importance(model_name, model, X_test, y_test, scoring_func=mean_squared_error, n_repeats=100):
    """
    Computes permutation-based feature importance for a model.
    """
    if model_name == 'AutoGluon':
        test_data = pd.DataFrame(X_test, columns=feature_names)
        test_data[target] = y_test
        baseline_score = scoring_func(y_test, model.predict(test_data).values)
    elif model_name == 'H2O':
        test_data = pd.DataFrame(X_test, columns=feature_names)
        test_data[target] = y_test
        test_df = h2o.H2OFrame(test_data)
        baseline_score = scoring_func(y_test, model.predict(test_df).as_data_frame().values.ravel())
    else:
        baseline_score = scoring_func(y_test, model.predict(X_test))

    feature_importances = {feature: [] for feature in range(X_test.shape[1])}

    for feature_idx in range(X_test.shape[1]):
        for _ in range(n_repeats):
            X_test_shuffled = X_test.copy()
            np.random.shuffle(X_test_shuffled[:, feature_idx])

            if model_name == 'AutoGluon':
                test_data = pd.DataFrame(X_test_shuffled, columns=feature_names)
                test_data[target] = y_test
                shuffled_score = scoring_func(y_test, model.predict(test_data).values)
            elif model_name == 'H2O':
                test_data = pd.DataFrame(X_test_shuffled, columns=feature_names)
                test_data[target] = y_test
                test_df = h2o.H2OFrame(test_data)
                shuffled_score = scoring_func(y_test, model.predict(test_df).as_data_frame().values.ravel())
            else:
                shuffled_score = scoring_func(y_test, model.predict(X_test_shuffled))

            feature_importances[feature_idx].append(shuffled_score - baseline_score)

    return {feature: np.mean(importances) for feature, importances in feature_importances.items()}


# ==============================
# Evaluation Loop
# ==============================

for run in range(n_runs):
    seed = run * 37 + 1001
    test_size = 0.30

    datasets = [
        read_yuan(dataset='RCA', target='CS', test_size=test_size, seed=seed, plot=False),
        read_yuan(dataset='RCA', target='FS', test_size=test_size, seed=seed, plot=False),
    ]

    for dataset in datasets:
        dr = dataset['name'].replace(' ', '_').replace("'", "").lower()
        path = './json_automl_' + dr + '/'
        os.system('mkdir -p ' + path)

        for tk, tn in enumerate(dataset['target_names']):
            print(tk, tn)
            dataset_name = dataset['name'] + '-' + tn
            target = dataset['target_names'][tk]
            y_train, y_test = dataset['y_train'][tk], dataset['y_test'][tk]
            X_train, X_test = dataset['X_train'], dataset['X_test']
            n_samples_train, n_features = dataset['n_samples'], dataset['n_features']
            task, normalize = dataset['task'], dataset['normalize']
            feature_names = dataset['feature_names']
            n_samples_test = len(y_test)

            print('=' * 80)
            print(f'Dataset                    : {dataset_name}')
            print(f'Output                     : {tn}')
            print(f'Number of training samples : {n_samples_train}')
            print(f'Number of testing  samples : {n_samples_test}')
            print(f'Number of features         : {n_features}')
            print(f'Normalization              : {normalize}')
            print(f'Task                       : {task}')
            print('=' * 80)

            scoring = 'f1_micro' if task == 'classification' else 'neg_root_mean_squared_error'

            train_data = pd.DataFrame(X_train, columns=feature_names)
            train_data[target] = y_train
            test_data = pd.DataFrame(X_test, columns=feature_names)
            test_data[target] = y_test

            train_df = h2o.H2OFrame(train_data)
            test_df = h2o.H2OFrame(test_data)

            flaml = AutoML()
            flaml_settings = {
                "time_budget": time_budget,
                "metric": "mae",
                "task": "regression",
                "log_file_name": "ucs.log",
                "estimator_list": ['lgbm', 'rf', 'xgboost', 'extra_tree', 'xgb_limitdepth', 'sgd', 'kneighbor', 'histgb'],
                "seed": seed,
                "verbose": False,
            }

            pipeline_optimizer = TPOTRegressor(generations=20, population_size=20, cv=5, random_state=seed, verbosity=0)

            include = {
                'regressor': ['extra_trees', 'gaussian_process', 'gradient_boosting',
                              'k_nearest_neighbors', 'liblinear_svr', 'libsvm_svr',
                              'mlp', 'random_forest', 'sgd'],
                'feature_preprocessor': ['no_preprocessing']
            }

            for auto in ['AutoGluon', 'AutoKeras', 'H2O', 'TPOT', 'FLAML']:
                start_time = time.time()
                print(auto)

                if auto == 'FLAML':
                    automl = flaml
                    automl.fit(X_train=X_train, y_train=y_train, **flaml_settings)
                    y_pred = automl.predict(X_test)

                elif auto == 'TPOT':
                    automl = pipeline_optimizer
                    automl.fit(X_train, y_train)
                    y_pred = automl.predict(X_test)

                elif auto == 'AutoGluon':
                    automl = TabularPredictor(label=target).fit(train_data=train_data, verbosity=False)
                    y_pred = automl.predict(test_data).values

                elif auto == 'AutoKeras':
                    automl = StructuredDataRegressor(max_trials=50, column_names=list(feature_names),
                                                     loss='mean_absolute_error', seed=seed)
                    automl.fit(x=X_train, y=y_train, epochs=epochs, verbose=False)
                    y_pred = automl.predict(X_test).ravel()

                elif auto == 'H2O':
                    automl = H2OAutoML(max_runtime_secs=time_budget, seed=seed, sort_metric='rmse')
                    automl.train(x=list(feature_names), y=target, training_frame=train_df)
                    y_pred = automl.predict(test_df).as_data_frame().values.ravel()

                elapsed_time = time.time() - start_time
                print('>> ', run, auto, '		', r2_score(y_test, y_pred), elapsed_time)

                feature_importances = permutation_feature_importance(auto, automl, X_test, y_test, scoring_func=mean_squared_error, n_repeats=10)

                results = [{
                    'run': run,
                    'elapsed_time': elapsed_time,
                    'seed': seed,
                    'estimator': auto,
                    'y_pred': y_pred.tolist(),
                    'y_test': y_test.tolist(),
                    'r2': r2_score(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'dataset': dataset_name,
                    'target': target,
                    'feature_names': feature_names.tolist(),
                    'feature_importances': list(feature_importances.values()),
                }]

                filename = (
                    f"{path}{basename}_run_{run:02d}_"
                    f"{dataset_name:15s}_{auto:11s}_{target:15s}.json"
                ).replace(' ', '_').replace("'", "").lower()

                with open(filename, 'w') as fp:
                    json.dump(results, fp)
