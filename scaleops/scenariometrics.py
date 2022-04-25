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
                 query: String,
                 metric: Union[Matrix, Vector, Scalar, String]):
        self.query = query
        self.metric = metric


class ScenarioMetrics:

    def __init__(self,
                 prometheus: promqlpandas.Prometheus):
        self.prometheus = prometheus

    def _query_range(self, query: str, scenario: Scenario,
                     labels: Dict = None, flush_cache: bool = False) -> Matrix:
        return self.prometheus.query_range(query, scenario.start,
                                           scenario.end,
                                           scenario.step,
                                           labels,
                                           flush_cache=flush_cache)


class PrometheusMetricsMixin(ScenarioMetrics):

    def scrape_duration(self, scenario: Scenario, flush_cache: bool = False) -> \
            List[MetricResult]:
        scrape_duration_q = f"""scrape_duration_seconds{{
    kubernetes_pod_name=~".*{scenario.pod_part}.*"
}}"""
        return [MetricResult(scrape_duration_q,
                             self._query_range(scrape_duration_q, scenario,
                                               flush_cache=flush_cache))]


class JvmScenarioMetricsMixin(ScenarioMetrics):

    def jvm_threads(self, scenario: Scenario, flush_cache: bool = False) -> \
            List[MetricResult]:
        jvm_threads_current_q = f"""jvm_threads_current{{
    kubernetes_pod_name=~".*{scenario.pod_part}.*"
}} > 0"""
        return [MetricResult(jvm_threads_current_q,
                             self._query_range(jvm_threads_current_q,
                                               scenario,
                                               flush_cache=flush_cache))]

    def jvm_memory(self, scenario: Scenario, flush_cache: bool = False) -> List[
        MetricResult]:
        jvm_mem_bytes_committed_q = f"""jvm_memory_bytes_committed{{
    kubernetes_pod_name=~".*{scenario.pod_part}.*"
}} > 0"""
        jvm_mem_bytes_used_q = f"""jvm_memory_bytes_used{{
    kubernetes_pod_name=~".*{scenario.pod_part}.*"
}} > 0"""
        jvm_mem_pool_bytes_used_q = f"""jvm_memory_pool_bytes_used{{
    kubernetes_pod_name=~".*{scenario.pod_part}.*"
}} > 0"""
        jvm_mem_pool_bytes_committed_q = f"""jvm_memory_pool_bytes_committed{{
    kubernetes_pod_name=~".*{scenario.pod_part}.*"
}} > 0"""

        jvm_mem_bytes_committed_df = self._query_range(
                jvm_mem_bytes_committed_q, scenario, flush_cache=flush_cache)

        jvm_mem_bytes_used_df = self._query_range(jvm_mem_bytes_used_q,
                                                  scenario,
                                                  flush_cache=flush_cache)
        jvm_mem_pool_bytes_committed_df = self._query_range(
                jvm_mem_pool_bytes_committed_q, scenario,
                flush_cache=flush_cache)

        jvm_mem_pool_bytes_used_df = self._query_range(
                jvm_mem_pool_bytes_used_q, scenario, flush_cache=flush_cache)

        return [MetricResult(jvm_mem_bytes_committed_q,
                             jvm_mem_bytes_committed_df),
                MetricResult(jvm_mem_bytes_used_q,
                             jvm_mem_bytes_used_df),
                MetricResult(jvm_mem_pool_bytes_committed_q,
                             jvm_mem_pool_bytes_committed_df),
                MetricResult(jvm_mem_pool_bytes_used_q,
                             jvm_mem_pool_bytes_used_df)]

    def jvm_gc(self, scenario: Scenario, flush_cache: bool = False) -> List[
        MetricResult]:
        jvm_gc_collection_seconds_sum_q = f"""rate(jvm_gc_collection_seconds_sum{{
    kubernetes_pod_name=~".*{scenario.pod_part}.*"
}}[1m])"""
        jvm_gc_collection_seconds_count_q = f"""rate(jvm_gc_collection_seconds_count{{
    kubernetes_pod_name=~".*{scenario.pod_part}.*"
}}[1m])"""

        jvm_gc_sum_df = self._query_range(
                jvm_gc_collection_seconds_sum_q,
                scenario,
                {
                    'metric_name': 'jvm_gc_collection_seconds_sum_1m'},
                flush_cache=flush_cache
        )
        jvm_gc_count_df = self._query_range(
                jvm_gc_collection_seconds_count_q,
                scenario,
                {
                    'metric_name': 'jvm_gc_collection_seconds_count_1m'},
                flush_cache=flush_cache
        )

        return [MetricResult(jvm_gc_collection_seconds_sum_q, jvm_gc_sum_df),
                MetricResult(jvm_gc_collection_seconds_count_q,
                             jvm_gc_count_df)]


class ContainerScenarioMetricsMixin(ScenarioMetrics):

    def cluster_size(self, scenario: Scenario, flush_cache: bool = False) -> \
            List[MetricResult]:
        kube_container_info_q = f'kube_pod_container_info{{pod=~".*{scenario.pod_part}.*"}}'
        return [MetricResult(kube_container_info_q,
                             self.prometheus.query(kube_container_info_q,
                                                   time=scenario.end,
                                                   flush_cache=flush_cache))]

    def container_cpu_utilization(self, scenario: Scenario,
                                  flush_cache: bool = False) -> List[
        MetricResult]:
        container_cpu_usage_seconds_total_q = f"""sum(rate(container_cpu_usage_seconds_total{{
    pod=~".*{scenario.pod_part}.*",
    container!~"POD",
    image!=""
}}[1m])) by (pod)
/
sum(container_spec_cpu_quota{{
    pod=~".*{scenario.pod_part}.*",
    container!~"POD",
    image!=""
}}) by (pod) * 100000"""
        return [MetricResult(container_cpu_usage_seconds_total_q,
                             self._query_range(
                                     container_cpu_usage_seconds_total_q,
                                     scenario,
                                     {
                                         'metric_name': 'container_cpu_utilization'},
                                     flush_cache=flush_cache
                             ))]

    def disk_bytes(self, scenario, flush_cache: bool = False) -> List[
        MetricResult]:
        container_fs_read_bytes_total_q = f"""rate(container_fs_reads_bytes_total{{
    pod=~".*{scenario.pod_part}.*"
}}[1m])"""
        container_fs_write_bytes_total_q = f"""rate(container_fs_writes_bytes_total{{
    pod=~".*{scenario.pod_part}.*"
}}[1m])"""

        disk_bytes_reads_df = self._query_range(
                container_fs_read_bytes_total_q,
                scenario, {
                    'metric_name': 'container_fs_reads_bytes_total_1m'},
                flush_cache=flush_cache
        )
        disk_bytes_writes_df = self._query_range(
                container_fs_write_bytes_total_q,
                scenario, {
                    'metric_name': 'container_fs_writes_bytes_total_1m'},
                flush_cache=flush_cache
        )

        return [
            MetricResult(container_fs_read_bytes_total_q, disk_bytes_reads_df),
            MetricResult(container_fs_write_bytes_total_q,
                         disk_bytes_writes_df)]

    def disk_io(self, scenario: Scenario, flush_cache: bool = False) -> List[
        MetricResult]:
        container_fs_reads_total_q = f"""rate(container_fs_reads_total{{
    pod=~".*{scenario.pod_part}.*",
}}[1m])"""
        container_fs_writes_total_q = f"""rate(container_fs_writes_total{{
    pod=~".*{scenario.pod_part}.*",
}}[1m])"""

        disk_io_reads = self._query_range(
                container_fs_reads_total_q, scenario,
                {
                    'metric_name': 'container_fs_reads_total_1m'},
                flush_cache=flush_cache
        )
        disk_io_writes = self._query_range(
                container_fs_writes_total_q,
                scenario,
                {
                    'metric_name': 'container_fs_writes_total_1m'},
                flush_cache=flush_cache
        )

        return [MetricResult(container_fs_reads_total_q, disk_io_reads),
                MetricResult(container_fs_writes_total_q, disk_io_writes)]

    def network_bytes(self, scenario: Scenario, flush_cache: bool = False) -> \
            List[MetricResult]:
        container_network_receive_bytes_total_q = f"""rate(container_network_receive_bytes_total{{
    pod=~".*{scenario.pod_part}.*"
}}[1m])"""
        container_network_transmit_bytes_total_q = f"""rate(container_network_transmit_bytes_total{{
    pod=~".*{scenario.pod_part}.*"
}}[1m])"""

        network_bytes_reads = self._query_range(
                container_network_receive_bytes_total_q,
                scenario, {
                    'metric_name': 'container_network_receive_bytes_total_1m'},
                flush_cache=flush_cache
        )
        network_bytes_writes = self._query_range(
                container_network_transmit_bytes_total_q,
                scenario, {
                    'metric_name': 'container_network_transmit_bytes_total_1m'},
                flush_cache=flush_cache
        )

        return [MetricResult(container_network_receive_bytes_total_q,
                             network_bytes_reads),
                MetricResult(container_network_transmit_bytes_total_q,
                             network_bytes_writes)]


class NodeScenarioMetricsMixin(ScenarioMetrics):
    def node_cpu_utilization(self, scenario: Scenario,
                             flush_cache: bool = False) -> List[MetricResult]:
        node_cpu_seconds_total_q = "rate(node_cpu_seconds_total[1m])"
        return [MetricResult(node_cpu_seconds_total_q, self._query_range(
                node_cpu_seconds_total_q,
                scenario,
                {'metric_name': 'node_cpu_utilization'},
                flush_cache=flush_cache
        ))]
