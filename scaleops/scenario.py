import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

Timestamp = Union[
    str, float, datetime.datetime]  # RFC-3339 string or as a Unix timestamp in seconds
Duration = Union[
    str, float, int, datetime.timedelta]  # Prometheus' duration string

__all__ = ['QueryTemplate', 'Scenario']


class QueryTemplate:
    """
    A QueryTemplate is a simple data class containing the elements needed to
    generate a concrete query.
    """

    def __init__(self, name: str, template_str: str,
                 template_params: Optional[Dict] = None,
                 labels: Optional[Dict] = None):
        self.name = name
        self.template_str = template_str
        self.template_params = template_params
        self.labels = labels


class Scenario:
    """
    A Scenario contains the elements needed to query a metric source, and
    enrich the results with additional labels.
    """

    def __init__(self, name: str, env: str,
                 start: Timestamp, end: Timestamp, step: Duration,
                 cache_path: Optional[Path] = None,
                 query_templates: Optional[List[QueryTemplate]] = None,
                 scenario_params: Optional[Dict] = None,
                 scenario_labels: Optional[Dict] = None):
        """
        :param name: the name of the Scenario
        :param env: the name of the environment where the scenario was run.
        :param start: the start time in a range query.
        :param end: the end time a range query.
        :param step: the step duration of a query.
        :param query_templates: a list of `QueryTemplate`s.
        :param scenario_params: a dictionary of scenario-level parameters for
                                use in the `query_templates`.
        :param scenario_labels: a dictionary of scenario-level labels for use in
                                all received metrics. Always includes the
                                scenario name in the labels.
        """

        self.name = name
        self.env = env
        self.start = start
        self.end = end
        self.step = step
        self.cache_path = cache_path
        self.query_templates = query_templates
        self.scenario_params = scenario_params
        if scenario_labels:
            self.scenario_labels = {'scenario': name} | scenario_labels
        else:
            self.scenario_labels = {'scenario': name}


def cv(x: np.ndarray):
    """
    Single variable Coefficient of Variation.

    https://en.wikipedia.org/wiki/Coefficient_of_variation

    :param x: The sample data.
    :return: The Coefficient of Variation of the sample.
    """
    return np.std(x) / np.nanmean(x)


def cv_rmsd(x: np.ndarray, y: np.ndarray):
    """
    Two variable Coefficient of Variation. Often `x` is the standard and `y` is
    the sample being compared.

    https://en.wikipedia.org/wiki/Coefficient_of_variation

    :param x: The base or standard data.
    :param y: The sample data.
    :return: The Coefficient of Variation between two samples.
    """
    diff = np.array(x) - np.array(y)
    n = len(x)
    return np.sqrt((diff * diff).sum() / n)


def perfect_throughput(n, l):
    """
    Formula for perfect Throughput from Concurrency/Load, where speedup is
    unbounded by serialization or crosstalk.

    :param n: Concurrency or load.
    :param l: Speedup - usually referred to as `lambda`.
    :return: Throughput at that concurrency/load and with `l` speedup.
    """
    return (l * n) / 1


def amdahl_throughput(n, l, s):
    """
    Formula for Amdahl's Law for Throughput from Concurrency/Load, which
    governs throughput when bounded by serialization but not crosstalk.

    :param n: Concurrency or load.
    :param l: Speedup - usually referred to as `lambda`.
    :param s: Serialization penalty - usually referred to as `sigma`.
    :return: Throughput at the concurrency/load with `l` speedup and `s`
             serialization cost.
    """
    return (l * n) / (1 + (s * (n - 1)))


def amdahl_response_time(x, l, s):
    """
    Formula for Amdahl's Law for Response Time from Throughput, which governs
    Response Time when bounded by serialization but not crosstalk.

    :param x: Throughput.
    :param l: Speedup - usually referred to as `lambda`.
    :param s: Serialization penalty - usually referred to as `sigma`.
    :return: Response Time at that Throughput with `l` speedup and `s`
             serialization cost.
    """
    return (s - 1) / (s * x - l)


def usl_throughput(n, l, s, k):
    """
    The Universal Scalability Law (USL) Formula for Throughput from
    Concurrency/Load, which governs throughput when bounded by both
    serialization and crosstalk.

    :param n: Concurrency or Load.
    :param l: Speedup - usually referred to as `lambda`.
    :param s: Serialization penalty - usually referred to as `sigma`.
    :param k: Crosstalk penalty - usually referred to as `kappa`.
    :return: Throughput at the concurrency/load with speedup `l`, `s`
             serialization cost, and `k` crostalk cost.
    """
    return (l * n) / (1 + (s * (n - 1)) + (k * n * (n - 1)))


def usl_response_time(n, l, s, k):
    """
    The Universal Scalability Law (USL) Formula for Response Time from
    Concurrency/Load, which governs response time when bounded by both
    serialization and crosstalk.

    :param n: Concurrency or Load.
    :param l: Speedup - usually referred to as `lambda`.
    :param s: Serialization penalty - usually referred to as `sigma`.
    :param k: Crosstalk penalty - usually referred to as `kappa`.
    :return: Throughput at the concurrency/load with speedup `l`, `s`
             serialization cost, and `k` crostalk cost.
    """
    return (1 + s * (n - 1) + k * n * (n - 1)) / l


def usl_throughput_by_response_time(r, l, s, k):
    """
    The Universal Scalability Law (USL) Formula for Throughput from Response
    Time, which governs throughput when bounded by both serialization and
    crosstalk.

    **NOTE: This is NOT a rational function! There are two solutions to this.**

    :param r: Response Time.
    :param l: Speedup - usually referred to as `lambda`.
    :param s: Serialization penalty - usually referred to as `sigma`.
    :param k: Crosstalk penalty - usually referred to as `kappa`.
    :return: Throughput at the concurrency/load with speedup `l`, `s`
             serialization cost, and `k` crosstalk cost.
    """
    return (np.sqrt(
            s ** 2 + k ** 2 + (2 * k * (2 * l * r + s - 2))) - k + s) / (
                   2 * k * r)


scrape_query_params = {
    'job': '.*',
    'instance': '.*'
}

scrape_query_templates = [
    QueryTemplate('up',
                  'up{job=~"${job}", instance=~"${instance}"}'),
    QueryTemplate('scrape_duration_seconds',
                  'scrape_duration_seconds{job=~"${job}", instance=~"${instance}"}'),
    QueryTemplate('scrape_samples_post_metric_relabeling',
                  'scrape_samples_post_metric_relabeling{job=~"${job}", instance=~"${instance}"}'),
    QueryTemplate('scrape_samples_scraped',
                  'scrape_samples_scraped{job=~"${job}", instance=~"${instance}"}'),
    QueryTemplate('scrape_series_added',
                  'scrape_series_added{job=~"${job}", instance=~"${instance}"}')
]
