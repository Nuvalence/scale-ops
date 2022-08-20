import datetime
import hashlib
import json
import logging
import os
import pathlib
import re
import shutil
import socket
from os.path import exists
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
import urllib3
from dateutil import parser as dtparser
from kubernetes import config
from kubernetes.client import Configuration
from kubernetes.client.api import core_v1_api
from kubernetes.stream import portforward

Timestamp = Union[
    str, float, datetime.datetime]  # RFC-3339 string or as a Unix timestamp in seconds
Duration = Union[
    str, float, int, datetime.timedelta]  # Prometheus duration string
Matrix = pd.DataFrame
Vector = pd.Series
Scalar = np.float64
String = str

# Get a logger object
logger = logging.getLogger(__name__)


class Prometheus:

    def __init__(self,
                 api_url: String,
                 headers: Optional[Dict] = None,
                 cache_path: Optional[pathlib.Path] = None,
                 k8s_context: Optional[String] = None):
        """
        Create a Prometheus Client.

        :param api_url: The URL to the Prometheus API endpoint.
        :param headers: Required headers for HTTP requests, default None.
        :param cache_path: Path to cache directory, default None.
        :param k8s_context: k8s context to use, default None.
        """
        logger.debug(f'Creating prometheus query object for {api_url}')
        self.api_url = api_url
        self.headers = headers
        self._cache_path = cache_path
        self._k8s_context = k8s_context
        if k8s_context:
            contexts, active_context = config.list_kube_config_contexts()
            if not contexts:
                raise RuntimeError(
                        "No Kubernetes contexts available in ~/.kube/config or $KUBECONFIG")
            contexts = [context['name'] for context in contexts]
            if k8s_context not in contexts:
                raise RuntimeError(
                        f'k8s_context="{k8s_context}" not found in ~/.kube/config or $KUBECONFIG.')

            config.load_kube_config(context=k8s_context)
            c = Configuration.get_default_copy()
            c.assert_hostname = False
            Configuration.set_default(c)
            self._core_v1 = core_v1_api.CoreV1Api()

    def __enter__(self):
        return self

    def label_values(self,
                     label: String,
                     query: Optional[String] = None,
                     start: Optional[Timestamp] = None,
                     end: Optional[Timestamp] = None) -> Vector:

        params = {}

        if start:
            epoch_start = to_ts(start)
            epoch_end = to_ts(end)
            params = {
                'start': epoch_start,
                'end': epoch_end,
            }
        if query:
            path = 'api/v1/series'
            params['match[]'] = query
        else:
            path = f'api/v1/label/{label}/values'

        results = self._do_query(path, params)

        if query:
            unique_values = pd.Series([r[label] for r in results]).unique()
        else:
            unique_values = pd.Series(np.array(results)).unique()

        return pd.Series(unique_values, name=label)

    def query(self,
              query: String,
              labels: Optional[Dict] = None,
              time: Optional[Timestamp] = None,
              timeout: Optional[Duration] = None,
              sort: Optional[Callable[[Dict], Any]] = None,
              flush_cache: Optional[bool] = False) -> Union[
        Matrix, Vector, Scalar, String]:
        """
        Evaluates an instant query at a single point in time.

        Uses the `/api/v1/series` endpoint.

        :param query: Prometheus expression query string.
        :param labels: A dictionary of labels to add to each set of metric labels. Optional.
        :param time: Evaluation timestamp. Optional.
        :param timeout: Evaluation timeout. Optional.
        :param sort: A function passed that generates a sort from a metric dictionary. Optional.
        :param flush_cache: Flush the cache before running the query. Optional.
        :return: Pandas DataFrame or Series.
        """
        params = {'query': query}

        if time is not None:
            params['time'] = to_ts(time)

        if timeout is not None:
            params['timeout'] = duration_to_s(timeout)

        query_hash = self._query_hash(params)

        df = self._read_and_return_cache(query_hash, flush_cache)
        if df:
            return df.loc[:, '0']

        results = self._do_query('api/v1/query', params)
        logger.debug(f'Received {len(results)} metrics')

        if sort:
            results = sorted(results, key=lambda r: sort(r['metric']))

        metric_series = self._to_pandas(results, labels=labels)
        if len(metric_series) > 0:
            metric_df = pd.DataFrame(metric_series)

            # make sure to write it if we're caching
            if self._cache_path:
                if exists(self._cache_path / f'{query_hash}.parquet'):
                    os.remove(self._cache_path / f'{query_hash}.parquet')
                metric_df.columns = metric_df.columns.astype(str)
                metric_df.to_parquet(
                        self._cache_path / f'{query_hash}.parquet',
                        use_deprecated_int96_timestamps=True
                )
        return metric_series

    def query_range(self,
                    query: String,
                    start: Timestamp,
                    end: Timestamp,
                    step: Duration,
                    labels: Optional[Dict] = None,
                    timeout: Optional[Duration] = None,
                    sort: Optional[Callable[[Dict], Any]] = None,
                    flush_cache: Optional[bool] = False) -> Matrix:
        """
        Evaluates an expression query over a range of time.

        :param query: Prometheus expression query string.
        :param start: Start timestamp.
        :param end: End timestamp.
        :param step: Query resolution step width in `duration` format or float number of seconds.
        :param labels: A dictionary of labels to add to each set of metric labels
        :param timeout: Evaluation timeout. Optional.
        :param sort: A function passed that generates a sort from a metric dictionary. Optional.
        :param flush_cache: Flush the query cache before calling.
        :return: Pandas DataFrame.
        """
        epoch_start = to_ts(start)
        epoch_end = to_ts(end)
        step_seconds = duration_to_s(step)
        params = {
            'query': query,
            'start': epoch_start,
            'end': epoch_end,
            'step': step_seconds
        }

        if timeout is not None:
            params['timeout'] = duration_to_s(timeout)

        query_hash = self._query_hash(params)

        df = self._read_and_return_cache(query_hash, flush_cache)
        if df:
            return df

        # get the data
        results = self._do_query('api/v1/query_range', params)
        metric_df = self._to_pandas(
                results,
                epoch_start,
                epoch_end,
                step_seconds,
                sort,
                labels)
        logger.debug(f'Received {len(results)} metrics')

        self._write_cache(metric_df, query_hash)
        return metric_df

    def _write_cache(self, metric_df, query_hash):
        # make sure to write it if we're caching
        if len(metric_df.index) > 0 and self._cache_path:
            if exists(self._cache_path / f'{query_hash}.parquet'):
                os.remove(self._cache_path / f'{query_hash}.parquet')
            metric_df.to_parquet(
                    self._cache_path / f'{query_hash}.parquet',
                    use_deprecated_int96_timestamps=True
            )

    def flush_cache(self) -> None:
        shutil.rmtree(self._cache_path, ignore_errors=True)

    def flush_query_cache(self, query_hash: String) -> None:
        # used for generating filenames for cache
        # noinspection InsecureHash\
        os.remove(self._cache_path / f'{query_hash}.parquet')

    def _do_query(self, path: str, params: Dict) -> Dict:
        if self._k8s_context:
            # Adapted from https://github.com/kubernetes-client/python/blob/master/examples/pod_portforward.py
            # Monkey patch the urllib3.util.connection.create_connection function so that
            # DNS names of the following formats will access kubernetes ports:
            #
            #    <pod-name>.<namespace>.kubernetes
            #    <pod-name>.pod.<namespace>.kubernetes
            #    <service-name>.svc.<namespace>.kubernetes
            #    <service-name>.service.<namespace>.kubernetes
            ##
            self._urllib3_create_connection = urllib3.util.connection.create_connection
            urllib3.util.connection.create_connection = self._kubernetes_create_connection

        with requests.Session() as http:
            resp = http.get(urljoin(self.api_url, path),
                            headers=self.headers,
                            params=params)
        if self._k8s_context:
            urllib3.util.connection.create_connection = self._urllib3_create_connection

        if resp.status_code not in [400, 422, 503]:
            resp.raise_for_status()

        response = resp.json()
        if response['status'] != 'success':
            raise RuntimeError('{errorType}: {error}'.format_map(response))

        return response['data']

    def _read_and_return_cache(self,
                               query_hash: String,
                               flush_cache: Optional[bool] = False) -> Optional[Matrix]:
        # don't use a cache if no path is set
        if self._cache_path:
            # if the cache_path was configured, and flush_cache is True
            if flush_cache:
                self.flush_query_cache(query_hash)
            # if the cache_path was configured, look there first
            if exists(self._cache_path):
                if exists(self._cache_path / f'{query_hash}.parquet'):
                    df = pd.read_parquet(
                            self._cache_path / f'{query_hash}.parquet')
                    return df
            else:
                os.makedirs(self._cache_path)
                return None

    @classmethod
    def _query_hash(cls, params: Dict) -> String:
        # used for generating filenames for cache
        # noinspection InsecureHash
        return hashlib.sha256(
                json.dumps(params, sort_keys=True).encode('utf-8')).hexdigest()

    @classmethod
    def _to_pandas(cls, results: Dict, start: float = None, end: float = None,
                   step: float = None,
                   sort: Optional[Callable[[Dict], Any]] = None,
                   labels: Dict = None) -> Union[
        Matrix, Vector, Scalar, String]:
        result_type = results['resultType']

        """Convert Prometheus data object to Pandas object."""
        if sort:
            r = sorted(results['result'], key=lambda r: sort(r['metric']))
        else:
            r = results['result']

        if result_type == 'vector':
            if len(r) > 0:
                return cls._numpy_to_series(
                        *cls._vector_to_numpy(r), labels=labels)
            return pd.Series()
        elif result_type == 'matrix':
            if len(r) > 0:
                return cls._numpy_to_dataframe(
                        *cls._matrix_to_numpy(r, start, end, step),
                        labels=labels)
            return pd.DataFrame()
        elif result_type == 'scalar':
            return np.float64(r)
        elif result_type == 'string':
            return r
        else:
            raise ValueError('Unknown type: {}'.format(result_type))

    @classmethod
    def _vector_to_numpy(cls, results: dict) -> Tuple[np.ndarray, List]:
        """Take a list of results and turn it into a numpy array."""

        # Create the destination array and NaN fill for missing data
        data = np.zeros((len(results)), dtype=np.float64)
        data[:] = np.nan

        metrics = []

        for ii, t in enumerate(results):
            metric = t['metric']
            value = t['value'][1]

            # Insert the data while converting all the string values data into floating
            # point, simply using `float` works fine as it supports all the string
            # prometheus uses for special values
            data[ii] = np.float64(value)

            metrics.append(metric)

        return data, metrics

    @classmethod
    def _matrix_to_numpy(cls, results: dict, start: float, end: float,
                         step: float) -> Tuple[
        np.ndarray, List, np.ndarray]:
        """Take a list of results and turn it into a numpy array."""

        # Calculate the full range of timestamps we want data at. Add a small constant
        # to the end to make it inclusive if it lies exactly on an interval boundary
        times = np.arange(start, end + 1e-6, step)

        # Create the destination array and NaN fill for missing data
        data = np.zeros((len(results), len(times)), dtype=np.float64)
        data[:] = np.nan

        metrics = []

        for ii, t in enumerate(results):
            metric = t['metric']
            metric_times, values = zip(*t['values'])

            # This identifies which slots to insert the data into. Note that it relies
            # on the fact that Prometheus produces the same grid of samples as we do in
            # here. That should be fine, and we use `np.rint` to mitigate any possible
            # rounding issues, but it's worth noting.
            inds = np.rint((np.array(metric_times) - start) / step).astype(
                    np.int64)

            # Insert the data while converting all the string values data into floating
            # point, simply using `float` works fine as it supports all the string
            # prometheus uses for special values
            data[ii, inds] = [np.float64(v) for v in values]

            metrics.append(metric)

        return data, metrics, times

    @classmethod
    def _numpy_to_series(cls, data: np.ndarray, metrics: List,
                         labels: Dict = None) -> Vector:
        index = cls._metric_index(metrics, labels)
        return pd.Series(data, index=index)

    @classmethod
    def _numpy_to_dataframe(cls, data: np.ndarray, metrics: List,
                            times: np.ndarray, labels: Dict = None) -> Matrix:
        columns = cls._metric_index(metrics, labels)
        index = pd.Index(pd.to_datetime(times, unit='s'), name='timestamp')
        return pd.DataFrame(data.T, columns=columns, index=index)

    @classmethod
    def _metric_index(cls, metrics: List, labels: Dict = None) -> pd.MultiIndex:
        # Merge labels if they exist
        metrics_labels = _merge_metric_labels(metrics, labels)
        # Get the set of all the unique label names
        levels = set()
        for m in metrics_labels:
            if '__name__' in m.keys():
                m['metric_name'] = m.pop('__name__')
            levels |= set(m.keys())
        levels = sorted(list(levels))
        if len(levels) == 0:
            raise RuntimeError(
                    'Queries that are constructed as pandas df need to have at least one label category in the results')

        # Get the set of label values for each metric series and turn into a multilevel index
        mt = [tuple(m.get(level, None) for level in levels) for m in
              metrics_labels]
        index = pd.MultiIndex.from_tuples(mt, names=levels)
        return index

    def _kubernetes_create_connection(
            self,
            address,
            timeout=socket.getdefaulttimeout(),
            source_address=None,
            socket_options=None,
    ):
        dns_name = address[0]
        if isinstance(dns_name, bytes):
            dns_name = dns_name.decode()
        dns_name = dns_name.split(".")
        if dns_name[-1] != 'kubernetes':
            return self._urllib3_create_connection(
                    address,
                    timeout,
                    source_address,
                    socket_options,
            )
        if len(dns_name) not in (3, 4):
            raise RuntimeError("Unexpected kubernetes DNS name.")
        namespace = dns_name[-2]
        name = dns_name[0]
        port = address[1]
        if len(dns_name) == 4:
            if dns_name[1] in ('svc', 'service'):
                service = self._core_v1.read_namespaced_service(name, namespace)
                for service_port in service.spec.ports:
                    if service_port.port == port:
                        port = service_port.target_port
                        break
                else:
                    raise RuntimeError(
                            "Unable to find service port: %s" % port)
                label_selector = []
                for key, value in service.spec.selector.items():
                    label_selector.append("%s=%s" % (key, value))
                pods = self._core_v1.list_namespaced_pod(
                        namespace, label_selector=",".join(label_selector)
                )
                if not pods.items:
                    raise RuntimeError("Unable to find service pods.")
                name = pods.items[0].metadata.name
                if isinstance(port, str):
                    for container in pods.items[0].spec.containers:
                        for container_port in container.ports:
                            if container_port.name == port:
                                port = container_port.container_port
                                break
                        else:
                            continue
                        break
                    else:
                        raise RuntimeError(
                                "Unable to find service port name: %s" % port)
            elif dns_name[1] != 'pod':
                raise RuntimeError(
                        "Unsupported resource type: %s" %
                        dns_name[1])
        pf = portforward(self._core_v1.connect_get_namespaced_pod_portforward,
                         name, namespace, ports=str(port))
        return pf.socket(port)


def _merge_metric_labels(metrics: List, labels: Dict) -> List:
    if labels:
        return [{**m, **labels} for m in metrics]

    return metrics


def to_ts(ts: Timestamp) -> int:
    """Convert a datetime or float to a UNIX timestamp.

    Parameters
    ----------
    ts
        If float assume it's already a UNIX timestamp, otherwise convert a
        datetime according to the usual Python rules.

    Returns
    -------
    timestamp
        UNIX timestamp.
    """
    if not isinstance(ts, (str, float, datetime.datetime)):
        raise TypeError(f'dt must be a float or datetime. Got {type(ts)}')

    if isinstance(ts, float):
        return int(round(ts))

    if isinstance(ts, str):
        return int(round(dtparser.parse(ts).timestamp()))

    return int(round(ts.timestamp()))


def duration_to_s(duration: Duration) -> float:
    """Convert a Prometheus duration string to an interval in s.

    Parameters
    ----------
    duration
        If float or in it is assumed to be in seconds, and is passed through.
        If a str it is parsed according to prometheus rules.

    Returns
    -------
    seconds
        The number of seconds corresponding to the duration.
    """
    if not isinstance(duration, (str, float, int)):
        raise TypeError(f'Cannot convert {duration}.')

    if isinstance(duration, (float, int)):
        return float(duration)

    if isinstance(duration, datetime.timedelta):
        return duration.total_seconds()

    duration_codes = {
        'ms': 0.001,
        's': 1,
        'm': 60,
        'h': 60 * 60,
        'd': 24 * 60 * 60,
        'w': 7 * 24 * 60 * 60,
        'y': 365 * 24 * 60 * 60,
    }

    # a single time component
    pattern = f'(\\d+)({"|".join(duration_codes.keys())})'

    if not re.fullmatch(f'({pattern})+', duration):
        raise ValueError(f'Invalid format of duration string {duration}.')

    seconds = 0
    for match in re.finditer(pattern, duration):
        num = match.group(1)
        code = match.group(2)

        seconds += int(num) * duration_codes[code]

    return seconds


def ts_to_ms(ts: Timestamp):
    return int(round(to_ts(ts) * 1000))
