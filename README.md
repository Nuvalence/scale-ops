# scale-ops - Scalability Operations

The intent of this repository is to both facilitate, and act as a record for, my recent efforts to
standardize scalability discovery, modeling and tuning.

Home to the following works in progress:

* [`scaleops`](src/scaleops) - A simple Python module for collecting a standardized set of metrics
  for the system under test (SUT).
* [Scaling Laws](scalibility-laws.ipynb) - A primer on the Universal Scalability Law (USL).
* [Example Scale Ops](example-scale-ops.ipynb) - An example analysis using these tools.

## Scalability as a function

[Gartner's definition of Scalability](gartner-scalability-definition):

> Scalability is the measure of a system’s ability to increase or decrease in performance and cost in response to changes in application and system processing demands.

While that is a sufficient definition for understanding what scalability is, it doesn't offer any
guidance on how you go about increasing or decreasing performance and cost based on processing
demands.

Systems that make up Computers, Computers, and systems of Computers are governed by the laws of
queueing theory, and queueing theory applies to all queues. Service time is service time whether
you're measuring it in a program or measuring it a retail checkout, and queue theory applies to
each.

[Dr. Neil J. Gunther defines scalability as a _mathematical
function_, a relationship between dependent and independent variables.](http://www.perfdynamics.com/Manifesto/USLscalability.html)

To understand a scalability model you must choose the correct variables to describe how the system
operates. Since scalability is driven by the _**work**_ done by the system, a few of those variables
could include:

* Units of work (requests)
* The rate of requests over time (arrival rate)
* The number of units of work in a system at a time (concurrency)
* The number of customers, users, or driver processes sending requests (actors)

Each of these play a role in the scalability function, depending on the system. For instance, it's
common to configure the number of nodes for a system under test (SUT) while holding constant the
number of requests each node handles, or if you're evaluating the JVM in a single pod, you would
vary the number of CPUs and maintain the same workload for each configuration. In the first case the
amount of work is defined by the number of requests applied to each node and the output is the
completion rate, and in the second case, the independent variable is the number of CPUs and the
dependent variable is the completion rate.

Fundamentally, scalability is a _function of size_ or _load_, which are both measures of _
concurrency_. The dependent variable will be the rate at which the system can complete work, or _
throughput_. In perfect scalability the system should complete more work as the size of the system
or the load on the system grows, so it should be an increasing function.

## Scalability as an operation

Scalability Operations are the combination of cultural philosophies, practices, and tools that
increase an organization's ability to deliver applications and services in an elastic manner.

_Or said another way, elasticity in the cloud is Scalability Operations in practice._

While cloud infrastructure and container orchestration support elasticity, taking advantage of that
elasticity becomes more and more challenging as systems grow in computational complexity.

_Or said another way, how do you know when to scale up, or scale down?_

Scalability Operations sets out to apply the science of scalability and queuing theory in a
practical and pragmatic fashion, with tools and examples of using those tools against real-world
data, for the purposes of answering the question of when to scale, and by how much.

## scaleops

A simple Python module for collecting, analysing, and identifying the performance characteristics of
a system under test.

### [promql_pandas.Prometheus](scaleops/promql_pandas.py)

Python library for querying [Prometheus](https://prometheus.io/) and accessing the results as
[Pandas](https://pandas.pydata.org/) data structures. Labels are converted to
a [MultiIndex](https://pandas.pydata.org/)

#### Example

Issuing an instant query at a single point in time produces a Vector as a pandas `Series`:

```python
from scaleops.promql_pandas import Prometheus
import time

p = Prometheus('http://dev.generic-k8s.io:9090/')
p.query('node_cpu_seconds_total{mode="system"}', time=time.time())
```

```shell
Out[26]: 
cpu  instance                       job   metric_name             mode  
0    demo.somehost.io:9100  node  node_cpu_seconds_total  system    768248.08
1    demo.somehost.io:9100  node  node_cpu_seconds_total  system    765672.38
dtype: float64
```

And evaluating an aggregate query over a time range produces a Matrix as a pandas `DataFrame`, with
a `TimeseriesIndex` for rows and a `MultiIndex` for columns:

```python
from scaleops.promql_pandas import Prometheus
from datetime import timedelta
import time

p = Prometheus('http://dev.generic-k8s.io:9090/')
p.query_range(
        'sum(rate(node_cpu_seconds_total{mode=~"system|user"}[1m])) by (mode)',
        start=time.time() - timedelta(hours=1).total_seconds(),
        end=time.time(),
        step='1m')
```

```shell
Out[31]: 
mode                             system      user
timestamp                                        
2022-03-23 13:23:46.908718080  0.014600  0.768600
2022-03-23 13:24:46.908718080  0.012401  0.754491
2022-03-23 13:25:46.908718080  0.013199  0.728527
2022-03-23 13:26:46.908718080  0.013398  0.734268
2022-03-23 13:27:46.908718080  0.011200  0.769800
                                 ...       ...
2022-03-23 14:19:46.908718080  0.012998  0.770261
2022-03-23 14:20:46.908718080  0.012200  0.791600
2022-03-23 14:21:46.908718080  0.011798  0.737882
2022-03-23 14:22:46.908718080  0.013195  0.791084
2022-03-23 14:23:46.908718080  0.015198  0.800872
```

### Scenarios, ScenarioSessions, and QueryTemplates

* [Scenario](promql_pandas/)

```python
from scaleops.scenario import QueryTemplate

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
```
