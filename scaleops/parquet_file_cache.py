import hashlib
import os
import pathlib
import types
from contextlib import ContextDecorator
from enum import Enum
from functools import wraps

import jsonpickle as jsonpickle
import pandas as pd


class ParquetCacheType(Enum):
    SERIES = 'series'
    DATAFRAME = 'dataframe'


# Caching decorator for DataFrame storage in Parquet files
class ParquetFileCache(ContextDecorator):

    def __init__(self,
                 cache_path=pathlib.Path(
                         '__file__').parent / '__promql_pandas_cache__',
                 cache_type=ParquetCacheType.DATAFRAME,
                 cache_reset=False):
        self.cache_path = cache_path
        self.cache_type = cache_type
        self.cache_reset = cache_reset
        os.makedirs(self.cache_path, exist_ok=True)

    def __call__(self, func):
        @wraps(func)
        def call_wrapper(*args, **kwargs):
            key = f'{ParquetFileCache.generate_query_key(*args)}'
            cache_file = self.cache_path / f'{key}-{self.cache_type.value}.parquet'
            if self.cache_reset or (
                    not cache_file.exists()):
                # resolved via the `functools.wraps` decorator
                # noinspection PyUnresolvedReferences
                result = func(*args, **kwargs)
                self.put(key, result)
            else:
                result = self.get(key)

            return result

        return call_wrapper

    def __get__(self, instance, cls):
        return self if instance is None else types.MethodType(self,
                                                              instance)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def put(self, key, result, cache_reset=False):
        cache_file = self.cache_path / f'{key}-{self.cache_type.value}.parquet'
        if cache_reset or self.cache_reset or (
                not cache_file.exists()):

            if (
                    self.cache_type == ParquetCacheType.SERIES and not isinstance(
                    result, pd.Series)) or (
                    self.cache_type == ParquetCacheType.DATAFRAME and not isinstance(
                    result, pd.DataFrame)):
                raise TypeError(
                        f"The result of computing {key} is not a {self.cache_type} result")

            if isinstance(result, pd.Series):
                df = pd.DataFrame(result)
                df.columns = df.columns.astype(str)
                df.to_parquet(
                        cache_file,
                        use_deprecated_int96_timestamps=True
                )
            else:
                result.to_parquet(
                        cache_file,
                        use_deprecated_int96_timestamps=True
                )

    def get(self, key):
        cache_file = self.cache_path / f'{key}-{self.cache_type.value}.parquet'
        if cache_file.exists():
            result = pd.read_parquet(cache_file, engine='fastparquet')
            if self.cache_type == ParquetCacheType.SERIES:
                result = result.iloc[:, 0]
            print(f'Found {len(result.index)} cached records')
        else:
            if self.cache_type == ParquetCacheType.SERIES:
                result = pd.Series([], dtype=pd.StringDtype())
            else:
                result = pd.DataFrame()
        return result

    @staticmethod
    def generate_query_key(*args):
        """
        Generates a unique key based on the hashed values of all the passed
        arguments. This makes a pretty bold assumption that the hash() function
        is deterministic, which is (probably) implementation specific.
        """
        from scaleops.scenario import QueryTemplate, Scenario
        from scaleops.scenario_session import ScenarioSession

        hashed_args = []
        for arg in args:
            if isinstance(
                    arg,
                    int
            ) or isinstance(
                    arg,
                    str
            ) or isinstance(
                    arg,
                    QueryTemplate
            ):
                hashed_args.append(arg)
            if isinstance(arg, Scenario):
                hashed_args.append(ParquetFileCache._hashable_scenario(arg))
            if isinstance(arg, ScenarioSession):
                hashed_args.append(
                        ParquetFileCache._hashable_scenario(arg.scenario))
        # this is md5 hashed again to avoid the key growing too large for memcached
        # noinspection InsecureHash
        return hashlib.md5(
                jsonpickle.encode(hashed_args).encode('utf-8')).hexdigest()

    @staticmethod
    def _hashable_scenario(scenario):
        from scaleops.scenario import Scenario
        if isinstance(scenario, Scenario):
            return Scenario(scenario.name, scenario.env,
                            scenario.start, scenario.end,
                            scenario.step,
                            scenario_params=scenario.scenario_params,
                            scenario_labels=scenario.scenario_labels)
        return None
