import datetime
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from scaleops import promqlpandas
from scaleops.scenario import Scenario

Timestamp = Union[
    str, float, datetime.datetime]  # RFC-3339 string or as a Unix timestamp in seconds
Duration = Union[
    str, float, int, datetime.timedelta]  # Prometheus' duration string
Matrix = pd.DataFrame
Vector = pd.Series
Scalar = np.float64
String = str


class MetricResult:

    def __init__(self,
                 scenario: Scenario,
                 query: String,
                 metric: Union[Matrix, Vector, Scalar, String]):
        self.scenario = scenario
        self.query = query
        self.metric = metric


class ScenarioMetrics:

    def __init__(self,
                 prometheus: promqlpandas.Prometheus):
        self.prometheus = prometheus

    def _query_range(self, query: str, scenario: Scenario,
                     labels: Dict = None,
                     flush_cache: bool = False) -> MetricResult:
        return MetricResult(scenario, query,
                            self.prometheus.query_range(query, scenario.start,
                                                        scenario.end,
                                                        scenario.step,
                                                        labels,
                                                        flush_cache=flush_cache))


class PrometheusMetricsMixin(ScenarioMetrics):

    def scrape_duration_seconds(self, scenario: Scenario,
                                flush_cache: bool = False) -> MetricResult:
        scrape_duration_q = f"""scrape_duration_seconds{{
    kubernetes_pod_name=~"{scenario.item_regex}"
}}"""
        return self._query_range(scrape_duration_q, scenario,
                                 flush_cache=flush_cache)


class JvmScenarioMetricsMixin(ScenarioMetrics):

    def jvm_threads_current(self, scenario: Scenario,
                            flush_cache: bool = False) -> \
            MetricResult:
        jvm_threads_current_q = f"""jvm_threads_current{{
    kubernetes_pod_name=~"{scenario.item_regex}"
}} > 0"""
        return self._query_range(jvm_threads_current_q,
                                 scenario,
                                 flush_cache=flush_cache)

    def jvm_memory_pool_bytes_committed(self, scenario: Scenario,
                                        flush_cache: bool = False) -> MetricResult:
        jvm_mem_pool_bytes_committed_q = f"""jvm_memory_pool_bytes_committed{{
            kubernetes_pod_name=~"{scenario.item_regex}"
        }} > 0"""

        return self._query_range(
                jvm_mem_pool_bytes_committed_q, scenario,
                flush_cache=flush_cache)

    def jvm_memory_pool_bytes_used(self, scenario: Scenario,
                                   flush_cache: bool = False) -> MetricResult:
        jvm_mem_pool_bytes_used_q = f"""jvm_memory_pool_bytes_used{{
            kubernetes_pod_name=~"{scenario.item_regex}"
        }} > 0"""

        return self._query_range(
                jvm_mem_pool_bytes_used_q, scenario,
                flush_cache=flush_cache)

    def jvm_memory_bytes_committed(self, scenario: Scenario,
                                   flush_cache: bool = False) -> MetricResult:
        jvm_mem_bytes_committed_q = f"""jvm_memory_bytes_committed{{
    kubernetes_pod_name=~"{scenario.item_regex}"
}} > 0"""

        return self._query_range(jvm_mem_bytes_committed_q, scenario,
                                 flush_cache=flush_cache)

    def jvm_memory_bytes_used(self, scenario: Scenario,
                              flush_cache: bool = False) -> \
            MetricResult:
        jvm_mem_bytes_used_q = f"""jvm_memory_bytes_used{{
            kubernetes_pod_name=~"{scenario.item_regex}"
        }} > 0"""

        return self._query_range(jvm_mem_bytes_used_q,
                                 scenario,
                                 flush_cache=flush_cache)

    def jvm_gc_collection_seconds_sum(self, scenario: Scenario,
                                      flush_cache: bool = False) -> MetricResult:
        jvm_gc_collection_seconds_sum_q = f"""rate(jvm_gc_collection_seconds_sum{{
            kubernetes_pod_name=~"{scenario.item_regex}"
        }}[1m])"""

        return self._query_range(
                jvm_gc_collection_seconds_sum_q,
                scenario,
                flush_cache=flush_cache
        )

    def jvm_gc_collection_seconds_count(self, scenario: Scenario,
                                        flush_cache: bool = False) -> MetricResult:
        jvm_gc_collection_seconds_count_q = f"""rate(jvm_gc_collection_seconds_count{{
    kubernetes_pod_name=~"{scenario.item_regex}"
}}[1m])"""

        return self._query_range(
                jvm_gc_collection_seconds_count_q,
                scenario,
                flush_cache=flush_cache
        )


class ContainerScenarioMetricsMixin(ScenarioMetrics):

    def kube_pod_container_info(self, scenario: Scenario,
                                flush_cache: bool = False) -> \
            MetricResult:
        kube_container_info_q = f'kube_pod_container_info{{pod=~"{scenario.item_regex}"}}'

        return MetricResult(scenario,
                            kube_container_info_q,
                            self.prometheus.query(kube_container_info_q,
                                     time=scenario.end,
                                     flush_cache=flush_cache))

    def container_cpu_usage_seconds_total(self, scenario: Scenario,
                                          flush_cache: bool = False) -> MetricResult:
        container_cpu_usage_seconds_total_q = f"""rate(container_cpu_usage_seconds_total{{
    pod=~"{scenario.item_regex}",
    container!~"POD",
    image!=""
}}[1m])"""

        return self._query_range(
                container_cpu_usage_seconds_total_q,
                scenario,
                flush_cache=flush_cache
        )

    def container_fs_reads_bytes_total(self, scenario,
                                       flush_cache: bool = False) -> MetricResult:
        container_fs_read_bytes_total_q = f"""rate(container_fs_reads_bytes_total{{
    pod=~"{scenario.item_regex}"
}}[1m])"""

        return self._query_range(
                container_fs_read_bytes_total_q,
                scenario,
                flush_cache=flush_cache
        )

    def container_fs_writes_bytes_total(self, scenario,
                                        flush_cache: bool = False) -> MetricResult:
        container_fs_write_bytes_total_q = f"""rate(container_fs_writes_bytes_total{{
    pod=~"{scenario.item_regex}",
    container!~"POD",
    image!=""
}}[1m])"""

        return self._query_range(
                container_fs_write_bytes_total_q,
                scenario,
                flush_cache=flush_cache
        )

    def container_fs_reads_total(self, scenario: Scenario,
                                 flush_cache: bool = False) -> MetricResult:
        container_fs_reads_total_q = f"""rate(container_fs_reads_total{{
    pod=~"{scenario.item_regex}",,
    container!~"POD",
    image!=""
}}[1m])"""

        return self._query_range(
                container_fs_reads_total_q,
                scenario,
                flush_cache=flush_cache
        )

    def container_fs_writes_total(self, scenario: Scenario,
                                  flush_cache: bool = False) -> MetricResult:
        container_fs_writes_total_q = f"""rate(container_fs_writes_total{{
    pod=~"{scenario.item_regex}",,
    container!~"POD",
    image!=""
}}[1m])"""

        return self._query_range(
                container_fs_writes_total_q,
                scenario,
                flush_cache=flush_cache
        )

    def container_network_receive_bytes_total(self, scenario: Scenario,
                                              flush_cache: bool = False) -> \
            MetricResult:
        container_network_receive_bytes_total_q = f"""rate(container_network_receive_bytes_total{{
    pod=~"{scenario.item_regex}",
    container!~"POD",
    image!=""
}}[1m])"""

        return self._query_range(
                container_network_receive_bytes_total_q,
                scenario,
                flush_cache=flush_cache
        )

    def container_network_transmit_bytes_total(self, scenario: Scenario,
                                               flush_cache: bool = False) -> \
            MetricResult:
        container_network_transmit_bytes_total_q = f"""rate(container_network_transmit_bytes_total{{
    pod=~"{scenario.item_regex}",
    container!~"POD",
    image!=""
}}[1m])"""

        return self._query_range(
                container_network_transmit_bytes_total_q,
                scenario,
                flush_cache=flush_cache
        )


class NodeScenarioMetricsMixin(ScenarioMetrics):
    def node_cpu_seconds_total(self, scenario: Scenario,
                               flush_cache: bool = False) -> MetricResult:
        node_cpu_seconds_total_q = f"""rate(node_cpu_seconds_total{{
    app=~"{scenario.item_regex}"
}}[1m])"""

        return self._query_range(
                node_cpu_seconds_total_q,
                scenario,
                flush_cache=flush_cache
        )

    def node_disk_read_bytes_total(self, scenario: Scenario,
                                   flush_cache: bool = False) -> MetricResult:
        node_disk_read_bytes_total_q = f"""rate(node_disk_read_bytes_total{{
    app=~"{scenario.item_regex}"
}}[1m])"""

        return self._query_range(
                node_disk_read_bytes_total_q,
                scenario,
                flush_cache=flush_cache
        )

    def node_disk_written_bytes_total(self, scenario: Scenario,
                                      flush_cache: bool = False) -> MetricResult:
        node_disk_written_bytes_total_q = f"""rate(node_disk_written_bytes_total{{
    app=~"{scenario.item_regex}"
}}[1m])"""

        return self._query_range(
                node_disk_written_bytes_total_q,
                scenario,
                flush_cache=flush_cache
        )

    def node_network_receive_bytes_total(self, scenario: Scenario,
                                         flush_cache: bool = False) -> \
            MetricResult:
        node_network_receive_bytes_total_q = f"""rate(node_network_receive_bytes_total{{
    app=~"{scenario.item_regex}"
}}[1m])"""

        return self._query_range(
                node_network_receive_bytes_total_q,
                scenario,
                flush_cache=flush_cache
        )

    def node_network_transmit_bytes_total(self, scenario: Scenario,
                                          flush_cache: bool = False) -> \
            MetricResult:
        node_network_transmit_bytes_total_q = f"""rate(node_network_transmit_bytes_total{{
    app=~"{scenario.item_regex}"
}}[1m])"""

        return self._query_range(
                node_network_transmit_bytes_total_q,
                scenario,
                flush_cache=flush_cache
        )
