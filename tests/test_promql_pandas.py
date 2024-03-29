import pprint
from datetime import datetime, timedelta
from unittest import TestCase

import pandas as pd

from scaleops.promql_pandas import Prometheus


class TestPrometheus(TestCase):
    def setUp(self) -> None:
        self._p = Prometheus('http://localhost:9090/')

    def test_query(self):
        result = self._p.query('node_exporter_build_info')
        pprint.pprint(result)
        self.assertIsInstance(result, pd.Series,
                              'result should be a pandas Series object')

    def test_query_range(self):
        result = self._p.query_range('rate(node_cpu_seconds_total[5m])',
                                     datetime.now() - timedelta(minutes=30),
                                     datetime.now() - timedelta(minutes=20),
                                     '1m',
                                     {
                                         'metric_name': 'node_cpu_seconds_total_5m'})
        pprint.pprint(result)
        pprint.pprint(result.columns)
        self.assertIsInstance(result, pd.DataFrame,
                              'result should be a pandas DataFrame object')
        self.assertIn('metric_name', result.columns.names)
        result2 = self._p.query_range('rate(node_cpu_seconds_total[5m])',
                                      datetime.now() - timedelta(minutes=30),
                                      datetime.now() - timedelta(minutes=20),
                                      '1m',
                                      {
                                          'metric_name': 'node_cpu_seconds_total_5m'})
        self.assertIsInstance(result2, pd.DataFrame,
                              'result2 should be a pandas DataFrame object')
        self.assertIn('metric_name', result2.columns.names)

    def test_label_values_by_name(self):
        result = self._p.label_values('job')
        pprint.pprint(result)
        pprint.pprint(result.index)
        self.assertEqual('job', result.name)
        self.assertIsInstance(result, pd.Series,
                              'result should be a pandas Series object')

    def test_label_values_by_metric(self):
        result = self._p.label_values('job', 'jvm_threads_current')
        pprint.pprint(result)
        pprint.pprint(result.index)
        self.assertEqual('job', result.name)
        self.assertIsInstance(result, pd.Series,
                              'result should be a pandas Series object')