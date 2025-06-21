# -*- coding: utf-8 -*-
"""
Utility script to read and preprocess the RCA dataset 
(Yuan et al., 2022) for regression tasks (CS or FS as target).

Includes optional data visualization and correlation heatmaps.

Source: https://www.mdpi.com/1996-1944/15/8/2823
"""

import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Compatibility with deprecated NumPy types
np.bool = np.bool_

# LaTeX-style plot configuration (optional)
pl.rc('text', usetex=True)
pl.rc('font', **{'family': 'serif', 'serif': ['Palatino']})

def read_yuan(target='CS', dataset=None, test_size=None, seed=None, plot=False):
    """
    Loads and preprocesses the RCA dataset for the given target.

    Parameters:
        target (str): Target variable to predict ('CS' or 'FS').
        dataset (str): Name of the dataset, used for labeling plots.
        test_size (float): Proportion of the data to be used as test set.
        seed (int): Random seed for reproducibility.
        plot (bool): If True, generates correlation heatmap and pair plots.

    Returns:
        dict: Dictionary containing processed dataset information.
    """
    # Load and reshape data
    fn = './data/data_yuan/yuan2022_recycled_aggregate_concrete.txt'
    X = np.loadtxt(fn).reshape(-1, 14)
    cols = 'weffc acr rcar pcs nmrcas nmnas bdrca bdna warca wana larca lana CS FS'.split(' ')
    X = pd.DataFrame(X, columns=cols)

    # Filter if FS is target
    if target == 'FS':
        X = X[X['FS'] != 0]

    target_names = [target]
    variable_names = list(X.columns.drop(target_names))
    X = X[variable_names + target_names]
    X.dropna(inplace=True)

    # Encode categorical columns if needed
    categorical_columns = []
    for cc in categorical_columns:
        le = LabelEncoder()
        le.fit(X[cc].values.ravel())
        X[cc] = le.transform(X[cc].values.ravel()).reshape(-1, 1)

    # Split data
    if not test_size:
        X_train, y_train = X[variable_names].values, X[target_names].values
        X_test, y_test = pd.DataFrame([[], ]).values, pd.DataFrame([[], ]).values
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X[variable_names].values, X[target_names].values,
            test_size=test_size, shuffle=True, random_state=seed)
        y_train, y_test = y_train.ravel(), y_test.ravel()

    df = X[variable_names + target_names].copy()

    if plot:
        # Plot correlation heatmap
        pl.figure(figsize=(5, 4))
        corr = df.corr().round(2)
        mask = np.triu(np.ones_like(corr, dtype=np.bool))
        heatmap = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
        heatmap.set_title(f'{dataset}: Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
        pl.savefig(f'{dataset}_heatmap_correlation.png', bbox_inches='tight', dpi=300)
        pl.show()

        # Define custom correlation dot plot for PairGrid
        def corrdot(*args, **kwargs):
            corr_r = args[0].corr(args[1], 'pearson')
            corr_text = f"{corr_r:2.2f}".replace("0.", ".")
            ax = plt.gca()
            ax.set_axis_off()
            marker_size = abs(corr_r) * 10000
            ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
                       vmin=-1, vmax=1, transform=ax.transAxes)
            font_size = abs(corr_r) * 40 + 5
            ax.annotate(corr_text, [.5, .5], xycoords="axes fraction",
                        ha='center', va='center', fontsize=font_size)

        # PairGrid plot with correlation dots
        sns.set(style='white', font_scale=1.6)
        g = sns.PairGrid(df, aspect=1.4, diag_sharey=False)
        g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'black'})
        g.map_diag(sns.histplot, kde=True)
        g.map_upper(corrdot)

    # Summary statistics to LaTeX
    n_samples, n_features = X_train.shape
    df_train = pd.DataFrame(X_train, columns=variable_names)
    df_train[target_names] = y_train.reshape(-1, 1)
    stat_train = df_train.describe().T
    stat_train.to_latex(buf=(dataset + '_train.tex').lower(), float_format="%.2f",
                        index=True, caption=f'Basic statistics for dataset {dataset}.')

    # Assemble result dictionary
    regression_data = {
        'task': 'regression',
        'name': dataset,
        'feature_names': np.array(variable_names),
        'target_names': target_names,
        'n_samples': n_samples,
        'n_features': n_features,
        'X_train': X_train,
        'y_train': y_train.reshape(1, -1),
        'X_test': X_test,
        'y_test': y_test.reshape(1, -1),
        'targets': target_names,
        'predicted_labels': None,
        'descriptions': 'None',
        'reference': "https://www.mdpi.com/1996-1944/15/8/2823",
        'items': None,
        'normalize': None,
    }

    return regression_data
