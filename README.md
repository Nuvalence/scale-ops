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

### promqlpandas.Prometheus

Python library for querying [Prometheus](https://prometheus.io/) and accessing the results as
[Pandas](https://pandas.pydata.org/) data structures. Labels are converted to
a [MultiIndex](https://pandas.pydata.org/)

#### Example

Issuing an instant query at a single point in time produces a Vector as a pandas `Series`:

```python
from scaleops.promqlpandas import Prometheus
import time

p = Prometheus('http://demo.robustperception.io:9090/')
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
from scaleops.promqlpandas import Prometheus
from datetime import timedelta
import time

p = Prometheus('http://demo.robustperception.io:9090/')
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

### [PrometheusScenarioMetricsMixin(ScenarioMetrics)](src/scaleops/scenariometrics.py)  

#### Scrape Duration

| Metric            | Source Measure | Source Type(s) | Result Type(s) | Description                                 | Join/Aggregate |
|-------------------|----------------|----------------|----------------|---------------------------------------------|---------------:|
| `scrape_duration` | seconds        | counter / gauge  | gauge          | average scrape duration in seconds over 1m  |        `False` |

### [ContainerScenarioMetricsMixin(ScenarioMetrics)](src/scaleops/scenariometrics.py)

#### CPU

| Metric                              | Source Measure | Source Type(s) | Result Type(s) | Description                 | Join/Aggregate |
|-------------------------------------|----------------|----------------|----------------|-----------------------------|---------------:|
| `container_cpu_usage_seconds_total` | CPU seconds    | counter        | gauge          | percent utilization over 1m |        `False` |

A very good description of CPU metrics in containers, and how to measure
them: [here](https://github.com/google/cadvisor/issues/2026#issuecomment-1003120833)

**Definitions**

* `container_cpu_usage_seconds_total` - CPU usage time in **seconds** of a specific container.
  Using `rate` on this metric will show how many CPU seconds were used per second.
* `container_spec_cpu_period` - The number of microseconds in a single CPU cycle by a single CPU
  unit. Typically 100000μs for docker.
* `container_spec_cpu_quota` - The number of total microseconds per CPU cycle, found by multiplying
  the number of CPU units by the `container_spec_cpu_period`. Only available if a CPU limit is
  present.

#### Disk - bytes

| Metric                            | Source Measure  | Source Type(s) | Result Type(s) | Description               | Join/Aggregate |
|-----------------------------------|-----------------|----------------|----------------|---------------------------|---------------:|
| `container_fs_reads_bytes_total`  | # bytes read    | counter        | gauge          | avg read bytes/sec in 1m  |        `False` |
| `container_fs_writes_bytes_total` | # bytes written | counter        | gauge          | avg write bytes/sec in 1m |        `False` |

#### Disk - io

| Metric                      | Source Measure  | Source Type(s) | Result Type(s) | Description               | Join/Aggregate |
|-----------------------------|-----------------|----------------|----------------|---------------------------|---------------:|
| `container_fs_reads_total`  | # bytes read    | counter        | gauge          | avg read bytes/sec in 1m  |        `False` |
| `container_fs_writes_total` | # bytes written | counter        | gauge          | avg write bytes/sec in 1m |        `False` |

#### Network - bytes

| Metric                                   | Source Measure  | Source Type(s) | Result Type(s) | Description               | Join/Aggregate |
|------------------------------------------|-----------------|----------------|----------------|---------------------------|---------------:|
| `container_network_receive_bytes_total`  | # bytes read    | counter        | gauge          | avg read bytes/sec in 1m  |        `False` |
| `container_network_transmit_bytes_total` | # bytes written | counter        | gauge          | avg write bytes/sec in 1m |        `False` |

### [JvmScenarioMetricsMixin(ScenarioMetrics)](src/scaleops/scenariometrics.py)

#### JVM Threads

| Metric                | Source Measure | Source Type(s) | Result Type(s) | Description     | Join/Aggregate |
|-----------------------|----------------|----------------|----------------|-----------------|---------------:|
| `jvm_threads_current` | # threads      | gauge          | gauge          | # threads in 1m |        `False` |

#### JVM Memory

| Metric                            | Source Measure         | Source Type(s) | Result Type(s) | Description   | Join/Aggregate |
|-----------------------------------|------------------------|----------------|----------------|---------------|---------------:|
| `jvm_memory_bytes_committed`      | # bytes committed      | gauge          | gauge          | # bytes in 1m |        `False` |
| `jvm_memory_bytes_used`           | # bytes used           | gauge          | gauge          | # bytes in 1m |        `False` |
| `jvm_memory_pool_bytes_committed` | # pool bytes committed | gauge          | gauge          | # bytes in 1m |        `False` |
| `jvm_memory_pool_bytes_used`      | # pool bytes used      | gauge          | gauge          | # bytes in 1m |        `False` |

#### JVM GC Time

| Metric                             | Source Measure   | Source Type(s) | Result Type(s)  | Description               | Join/Aggregate |
|------------------------------------|------------------|----------------|-----------------|---------------------------|---------------:|
| `jvm_gc_collection_seconds_sum`    | # seconds in GC  | counter        | gauge           | Sum of GC collection time |        `False` |
| `jvm_gc_collection_seconds_count`  | # of times in GC | counter        | gauge           | Count of GC collections   |        `False` |
