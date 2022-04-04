import pprint
from datetime import datetime, timedelta
from unittest import TestCase

import pandas as pd

from scaleops.promqlpandas import Prometheus


class TestPrometheus(TestCase):
    def setUp(self) -> None:
        self._p = Prometheus('http://demo.robustperception.io:9090/')

    def test_query(self):
        result = self._p.query('node_exporter_build_info')
        pprint.pprint(result)
        self.assertIsInstance(result, pd.Series, 'result should be a pandas Series object')

    def test_query_range(self):
        result = self._p.query_range('rate(node_cpu_seconds_total[5m])',
                                     datetime.now() - timedelta(days=2),
                                     datetime.now() - timedelta(days=1),
                                     '1m',
                                     {'metric_name': 'node_cpu_seconds_total_5m'})
        pprint.pprint(result)
        pprint.pprint(result.columns)
        self.assertIsInstance(result, pd.DataFrame, 'result should be a pandas DataFrame object')
        self.assertIn('metric_name', result.columns.names)
