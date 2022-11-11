import logging
import pathlib
import pprint
from datetime import datetime, timedelta
from unittest import TestCase

import pandas as pd

import scaleops
from scaleops.kube_port_forwarder import ShellKubePortForwarder
from scaleops.promql_pandas import Prometheus
from scaleops.scenario import Scenario
from scaleops.scenario_session import ScenarioSession


class ScenarioSessionTest(TestCase):
    _ss: ScenarioSession = None

    @classmethod
    def setUpClass(cls) -> None:
        prometheus = Prometheus('http://localhost:9090/')
        scenario = Scenario(
                name='test',
                env='dev',
                start=datetime.now() - timedelta(minutes=30),
                end=datetime.now() - timedelta(minutes=20),
                step='1m',
                query_templates=scaleops.scenario.scrape_query_templates,
                scenario_params=scaleops.scenario.scrape_query_params,
                cache_path=pathlib.Path(
                    '__file__').parent / '__promql_pandas_cache__'
        )
        kube_port_forwarder = ShellKubePortForwarder(
                kube_context='dev.generic-k8s.io',
                namespace='monitoring', name='prometheus-server', port=80,
                service=True)

        cls._up_scrape_query_template = \
            scaleops.scenario.scrape_query_templates[1]
        cls._ss = ScenarioSession(
                scenario=scenario,
                prometheus=prometheus,
                kube_port_forwarder=kube_port_forwarder,
                log_level=logging.DEBUG
        )
        cls._ss.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._ss.stop()

    def test_query_range(self):
        result = self._ss.query_range(self._up_scrape_query_template)
        pprint.pprint(result)
        pprint.pprint(result.columns)
        self.assertIsInstance(result, pd.DataFrame,
                              'result should be a pandas Series object')

    def test_label_values_by_name(self):
        result = self._ss.label_values('job')
        pprint.pprint(result)
        pprint.pprint(result.index)
        self.assertEqual('job', result.name)
        self.assertIsInstance(result, pd.Series,
                              'result should be a pandas Series object')

    def test_label_values_by_metric(self):
        result = self._ss.label_values('job', 'jvm_threads_current')
        pprint.pprint(result)
        pprint.pprint(result.index)
        self.assertEqual('job', result.name)
        self.assertIsInstance(result, pd.Series,
                              'result should be a pandas Series object')
