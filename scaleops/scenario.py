import datetime
from typing import Tuple, Union

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


def cv_rmsd(x: np.ndarray, y: np.ndarray):
    diff = np.array(x) - np.array(y)
    n = len(x)
    return np.sqrt((diff * diff).sum() / n)


def cv(x: np.ndarray):
    return np.std(x) / np.nanmean(x)


def plot_line_scenarios(metric_name: str, axis_name: str,
                        scenarios: list[Scenario],
                        metric_results: list[pd.DataFrame],
                        figsize: Tuple[int, int] = None):
    col_count = len(scenarios)
    fig, axs = plt.subplots(
            3,
            col_count,
            figsize=(18, 18),
            sharey='row'
    )
    for i, s in enumerate(scenarios):
        axs[0, i].plot(metric_results[i].iloc[:, [0]])
        axs[0, i].set_title(metric_results[i].iloc[:, [0]].columns[0])
        axs[2, i].plot(metric_results[i])
        axs[2, i].set_title(s.pod_part)

    for ax in axs.flat:
        ax.set(xlabel=metric_name, ylabel=axis_name)

    for ax in axs.flat:
        ax.label_outer()


def plot_cm_ab(metric_name: str,
               scenarios: list[Scenario],
               metric_results: list[pd.DataFrame],
               display_cm: bool = False,
               display_cv: bool = True,
               figsize: Tuple[int, int] = None):
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    col_count = len(scenarios)

    fig, axs = plt.subplots(
            1,
            col_count,
            figsize=(12.8, 9.6) if not figsize else figsize,
            sharex='all',
            sharey='all'
    )

    for i, s in enumerate(scenarios):
        cm = metric_results[i].corr(method=cv_rmsd)
        mask = np.triu(np.ones_like(cm, dtype=bool), k=0)

        if display_cm:
            display(cm)

        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        sns.heatmap(cm,
                    mask=mask,
                    cmap=cmap,
                    cbar=0,
                    cbar_ax=None if (col_count < i - 1) else cbar_ax,
                    ax=axs[i],
                    xticklabels=False,
                    yticklabels=False)
        axs[i].set_title(f'{s.pod_part} Correlation CV(RMSD): {metric_name}')

        if display_cv:
            m_a = metric_results[i].sum(axis=1)
            cv_a = cv(m_a)
            display(f'CV: {metric_name} {s.pod_part}: {cv_a}')


def scorecard(scenarios: list[Scenario],
              metric_results: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(
            [m.groupby(['metric_name'], axis=1).sum().agg(cv) for m in
             metric_results],
            axis=1,
            names=['scenario_name'],
            keys=[s.pod_part for s in scenarios]
    )
