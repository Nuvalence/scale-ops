import contextlib
import logging
import re
import sys
from contextlib import ContextDecorator
from string import Template
from typing import Optional, Union

import pandas as pd
from pandas import DataFrame

from scaleops.kube_port_forwarder import ShellKubePortForwarder
from scaleops.parquet_file_cache import ParquetCacheType, ParquetFileCache
from scaleops.promql_pandas import Prometheus
from scaleops.scenario import Duration, QueryTemplate, Scenario, Timestamp


class ScenarioSession(ContextDecorator):
    """
    Responsible for orchestrating the collection of metrics from the metric sources by
    interpreting the `Scenario` and the associated query templates into concrete
    queries and submitting those queries and collecting the results from
    Prometheus.
    """

    def __init__(self,
                 scenario: Scenario,
                 prometheus: Prometheus,
                 timeout: Optional[Timestamp] = None,
                 log_level: Union[int, str, None] = logging.INFO,
                 kube_port_forwarder: Optional[ShellKubePortForwarder] = None):
        """

        :param scenario:
        :param prometheus:
        :param timeout:
        :param log_level:
        """
        self.scenario = scenario
        self.prometheus = prometheus
        self.timeout = timeout
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)

        if kube_port_forwarder:
            self._kube_port_forwarder = kube_port_forwarder
        else:
            self._kube_port_forwarder = contextlib.nullcontext()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        if isinstance(self._kube_port_forwarder, ShellKubePortForwarder):
            self._kube_port_forwarder.start()

    def stop(self):
        if isinstance(self._kube_port_forwarder, ShellKubePortForwarder):
            self._kube_port_forwarder.stop()

    def get_scenario_metrics(self,
                             timeout: Optional[Duration] = None,
                             cache_reset: Optional[bool] = False) -> list[
        DataFrame]:
        """

        :param cache_reset:
        :param timeout:
        :return:
        """
        scenario_metrics = []

        self._logger.debug(
                f'Executing scenario {self.scenario.name} in {self.scenario.env}')
        for query_template in self.scenario.query_templates:
            self._logger.debug(
                    f'Executing query template {query_template.name}')
            scenario_metrics.append(
                    self.query_range(query_template, timeout,
                                     cache_reset=cache_reset))
            self._logger.debug(f'Received {query_template.name}')

        scenario_metrics = [sm for sm in scenario_metrics if not sm.empty]

        self._logger.debug(f'Received {len(scenario_metrics)} query results')
        return scenario_metrics

    def label_values(self,
                     label: str,
                     metric: Optional[str] = None,
                     cache_reset: Optional[bool] = False) -> pd.Series:
        """

        :param cache_reset:
        :param label:
        :param metric:
        :return:
        """
        cache_key = ParquetFileCache.generate_query_key(label, metric)
        cache = ParquetFileCache(self.scenario.cache_path,
                                 ParquetCacheType.SERIES)
        result = cache.get(cache_key)
        if result.empty:
            result = self.prometheus.label_values(
                    label,
                    metric
            )
        cache.put(cache_key, result, cache_reset)
        return result

    def query_range(self,
                    query_template: QueryTemplate,
                    timeout: Optional[Duration] = None,
                    cache_reset: Optional[bool] = False) -> pd.DataFrame:
        """

        :param cache_reset:
        :param query_template:
        :param timeout:
        :return:
        """
        template = Template(query_template.template_str)

        # template params override scenario params
        template_params = []
        if self.scenario.scenario_params and query_template.template_params:
            template_params = self.scenario.scenario_params | query_template.template_params
        elif self.scenario.scenario_params:
            template_params = self.scenario.scenario_params.copy()
        elif query_template.template_params:
            template_params = query_template.template_params.copy()

        for template_param in template_params:
            if 'label_values' in template_params[template_param]:
                template_param_value = template_params[template_param]
                label_values_match = re.search(
                        r'^label_values\((\w+)(,\s+(\w+.+))?\)',
                        template_param_value
                )
                if label_values_match:
                    # if possible, always use the cached results
                    label_values = self.label_values(
                            label=label_values_match.group(1),
                            metric=label_values_match.group(3)
                    )
                    # if we're using label_values, and nothing returns, default
                    # to NOT executing the query
                    if not label_values.empty:
                        template_params[
                            template_param] = template_param_value.replace(
                                label_values_match.group(),
                                f'({"|".join(label_values.values)})'
                        )
                    else:
                        return pd.DataFrame()

        query = template.substitute(template_params)

        # template labels override scenario labels
        labels = []
        if self.scenario.scenario_labels and query_template.labels:
            labels = self.scenario.scenario_labels | query_template.labels
        elif self.scenario.scenario_labels:
            labels = self.scenario.scenario_labels.copy()
        elif query_template.labels:
            labels = query_template.labels.copy()

        # always set the metric_name
        if 'metric_name' not in labels:
            labels['metric_name'] = query_template.name
        cache_key = ParquetFileCache.generate_query_key(query, self.scenario,
                                                        labels)
        cache = ParquetFileCache(self.scenario.cache_path)
        result = cache.get(cache_key)
        if result.empty:
            result = self.prometheus.query_range(
                    query,
                    self.scenario.start,
                    self.scenario.end,
                    self.scenario.step,
                    labels=labels,
                    timeout=timeout or self.timeout
            )
        cache.put(cache_key, result, cache_reset)
        return result
