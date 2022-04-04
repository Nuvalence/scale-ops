import datetime
from typing import Union

import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

Timestamp = Union[
    str, float, datetime.datetime]  # RFC-3339 string or as a Unix timestamp in seconds
Duration = Union[
    str, float, int, datetime.timedelta]  # Prometheus' duration string
Matrix = pd.DataFrame
Vector = pd.Series
Scalar = np.float64
String = str


class Scenario:

    def __init__(self, pod_part: String, group_part: String, start: Timestamp,
                 end: Timestamp, step: Duration):
        """
        :arg step: The step duration for the queries in this scenario.
        """
        self.pod_part = pod_part
        self.group_part = group_part
        self.start = start
        self.end = end
        self.step = step


def cv_rmsd(x, y):
    diff = np.array(x) - np.array(y)
    n = len(x)
    return np.sqrt((diff * diff).sum() / n)


def cv(x):
    return np.std(x) / np.nanmean(x)


def plot_line_ab(metric_name: str, axis_name: str, metric_result_a: pd.DataFrame,
            metric_result_b: pd.DataFrame):
    fig, axs = plt.subplots(3, 2, figsize=(18, 18), sharey='row')
    axs[0, 0].plot(metric_result_a.iloc[:, [0]])
    axs[0, 0].set_title(metric_result_a.iloc[:, [0]].columns[0])
    axs[0, 1].plot(metric_result_b.iloc[:, [0]])
    axs[0, 1].set_title(metric_result_b.iloc[:, [0]].columns[0])
    axs[1, 0].plot(metric_result_a.iloc[:, [1]])
    axs[1, 0].set_title(metric_result_a.iloc[:, [1]].columns[0])
    axs[1, 1].plot(metric_result_b.iloc[:, [1]])
    axs[1, 1].set_title(metric_result_b.iloc[:, [1]].columns[0])
    axs[2, 0].plot(metric_result_a)
    axs[2, 1].plot(metric_result_b)

    for ax in axs.flat:
        ax.set(xlabel=metric_name, ylabel=axis_name)

    for ax in axs.flat:
        ax.label_outer()


def plot_cm_ab(metric_name: str, name_a: str, name_b: str, metric_result_a: pd.DataFrame,
            metric_result_b: pd.DataFrame, display_cm: bool = False, display_cv: bool = True):
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    cm_a = metric_result_a.corr(method=cv_rmsd)
    mask_a = np.triu(np.ones_like(cm_a, dtype=bool), k=0)
    cm_b = metric_result_b.corr(method=cv_rmsd)
    mask_b = np.triu(np.ones_like(cm_b, dtype=bool), k=0)

    if display_cm:
        display(cm_a)
        display(cm_b)

    fig, axs = plt.subplots(1, 2, figsize=(12, 12), sharex='all', sharey='all')
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    sns.heatmap(cm_a, mask=mask_a, cmap=cmap, cbar=0, cbar_ax=None, ax=axs[0], xticklabels=False,
                yticklabels=False)
    axs[0].set_title(f'{name_a} Correlation CV(RMSD): {metric_name}')
    sns.heatmap(cm_b, mask=mask_b, cmap=cmap, cbar=1, cbar_ax=cbar_ax, ax=axs[1], xticklabels=False,
                yticklabels=False)
    axs[1].set_title(f'{name_b} Correlation CV(RMSD): {metric_name}')

    if display_cv:
        m_a = metric_result_a.sum(axis=1)
        m_b = metric_result_b.sum(axis=1)
        cv_a = cv(m_a)
        cv_b = cv(m_b)
        cv_a_to_b = cv_rmsd(m_a, m_b)
        cv_b_to_a = cv_rmsd(m_b, m_a)
        display(f'CV(RMSD): {metric_name} {name_a}: {cv_a}')
        display(f'CV(RMSD): {metric_name} {name_b}: {cv_b}')
        display(f'CV(RMSD): {metric_name} {name_a} to {name_b}: {cv_a_to_b}')
        display(f'CV(RMSD): {metric_name} {name_b} to {name_a}: {cv_b_to_a}')