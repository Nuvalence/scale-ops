import datetime
from typing import Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from ipywidgets import HTML
from matplotlib import pyplot as plt, ticker

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


def cv(x: pd.Series):
    return np.std(x) / np.nanmean(x)


def plot_line_scenarios(metric_name: str, axis_name: str,
                        scenarios: list[Scenario],
                        metric_results: list[pd.DataFrame],
                        figsize: Tuple[int, int] = None,
                        display_total: bool = False,
                        annotate_total: bool = False):
    col_count = len(scenarios)
    fig, axs = plt.subplots(
            3,
            col_count,
            figsize=(12.8, 9.6) if figsize is None else figsize,
            sharey='row'
    )
    formatter = ticker.EngFormatter(places=0)
    for i, s in enumerate(scenarios):
        # this is uncomfortable - there has to be a way to dynamically choose
        # and use and indexing strategy for slicing arrays, but I can't figure
        # it out right now. So I do it the old-fashioned way...
        if col_count == 1:
            axs[0].plot(metric_results[i].iloc[:, [0]])
            axs[0].set_title(metric_results[i].iloc[:, [0]].columns[0])
            axs[1].plot(metric_results[i].iloc[:, [1]])
            axs[1].set_title(metric_results[i].iloc[:, [1]].columns[0])
            axs[2].plot(metric_results[i])
            if display_total:
                _label_total(metric_results[i], formatter, axs[2])
            if annotate_total:
                _annotate_total(metric_results[i], formatter, axs[2])
            axs[2].set_title(s.pod_part)
        else:
            axs[0, i].plot(metric_results[i].iloc[:, [0]])
            axs[0, i].set_title(metric_results[i].iloc[:, [0]].columns[0])
            axs[1, i].plot(metric_results[i].iloc[:, [1]])
            axs[1, i].set_title(metric_results[i].iloc[:, [1]].columns[0])
            axs[2, i].plot(metric_results[i])
            if display_total:
                _label_total(metric_results[i], formatter, axs[2, i])
            if annotate_total:
                _annotate_total(metric_results[i], formatter, axs[2, i])
            axs[2, i].set_title(s.pod_part)

    for ax in axs.flat:
        ax.set(xlabel=metric_name, ylabel=axis_name)

    for ax in axs.flat:
        ax.label_outer()
        ax.yaxis.set_major_formatter(formatter)


def plot_cm_scenarios(metric_name: str,
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
    cbar_ax = fig.add_axes([.91, .3, .03, .4])

    for i, s in enumerate(scenarios):
        cm = metric_results[i].corr(method=cv_rmsd)

        mask = np.triu(np.ones_like(cm, dtype=bool), k=0)

        if display_cm:
            display(cm)

        if col_count == 1:
            ax = axs
        else:
            ax = axs[i]

        sns.heatmap(cm,
                    mask=mask,
                    cmap=cmap,
                    cbar=0 if (col_count < i - 1) else 1,
                    cbar_ax=None if (col_count < i - 1) else cbar_ax,
                    ax=ax,
                    xticklabels=False,
                    yticklabels=False)
        ax.set_title(f'{s.pod_part} Correlation CV(RMSD): {metric_name}')

        if display_cv:
            cv_a = metric_results[i].groupby(['metric_name'], axis=1).sum().agg(cv).iloc[0]
            if cv_a <= 1:
                display(HTML(f"""<div class="alert alert-block alert-success">
<b>Low Variance:</b> CV of {metric_name} for {s.pod_part} is <b>{cv_a:.2f}</b>
</div>"""))
            elif cv_a < 2.0:
                display(HTML(f"""<div class="alert alert-block alert-warning">
<b>High Variance:</b> CV of {metric_name} for {s.pod_part} is <b>{cv_a:.2f}</b>
</div>"""))
            else:
                display(HTML(f"""<div class="alert alert-block alert-danger">
<b>Danger:</b> CV of {metric_name} for {s.pod_part} is <b>{cv_a:.2f}</b>
</div>"""))


def scorecard(scenarios: list[Scenario],
              metric_results: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(
            [m.groupby(['metric_name'], axis=1).sum().agg(cv) for m in
             metric_results],
            axis=1,
            names=['scenario_name'],
            keys=[s.pod_part for s in scenarios]
    )


def _label_total(metric_result: pd.DataFrame, formatter: ticker.Formatter,
                 ax: plt.axis = None):
    total_df = metric_result.sum(axis=1)
    xmax = total_df.index[np.argmax(total_df)]
    ymax = total_df.max()
    text = 'total (max={}@{:%Y-%m-%d %H:%M:%S})'.format(
            formatter(ymax), xmax)
    ax.plot(total_df, label=text)
    ax.legend(framealpha=1, frameon=True)


def _annotate_total(metric_result: pd.DataFrame, formatter: ticker.Formatter,
                    ax: plt.axis = None):
    total_df = metric_result.sum(axis=1)
    xmax = total_df.index[np.argmax(total_df)]
    ymax = total_df.max()
    text = 'total (max={}@{:%Y-%m-%d %H:%M:%S})'.format(
            formatter(ymax), xmax)
    if not ax:
        ax = plt.gca()
    ax.plot(total_df)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrow_props = dict(arrowstyle="->",
                       connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrow_props, bbox=bbox_props, ha="left", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.04, 0.96), **kw)
    return ax
