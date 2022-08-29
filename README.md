## Synergy :  Looking Beyond GPUs for DNN Scheduling on Multi-Tenant Clusters

This repository contains the source code implementation of the OSDI paper "Looking Beyond GPUs for DNN Scheduling on Multi-Tenant Clusters". This work was done as part of Microsoft Research's Project Fiddle. This source code is available under the MIT License.

### Directory Structure

#### simulator
This contains code for the Synergy scheduler, that includes various scheduling policies (scheduler/), in a simulator harness (runner.py), and a deployment module using gRPC(deployment/).

#### src
This contains the src code for offline profiling alongside a detailed [README](README-offline-profiler.md) that discusses how to use it.

### Setup

#### Option 1

Use the docker image `jayashreemohan/synergy_dali`.
```
- docker pull jayashreemohan/synergy_dali:latest
- nvidia-docker run --ipc=host --mount src=/,target=/datadrive/,type=bind -it --rm --network=host --privileged jayashreemohan/synergy_dali:latest /bin/bash
- git clone https://github.com/msr-fiddle/synergy.git
- cd synergy/simulator/deployment
- make
```
Now skip to [Getting Started](README.md#getting-started)

#### Option 2 :  Building from scratch
Please install the following dependencies before proceeding. Tested on Python 3.6.8 and 3.8.10

#### Setup profiler
```
- cd src
- cd profiler; ./prereq.sh; cd ..
```

#### Setup simulator
```
- cd simulator/deployment
- ./upgrade_pip.sh
- pip install -r requirements.txt
- make
```

#### Setup iterator
Synergy uses its own PyTorch iterator that is built on top of DALI & CoorDL. So before you run any profiling experiment, or deployment in a real GPU cluster,  build a docker container with this iterator support by following these steps. Note that this is not required to run the simulation experiments.
```
- git clone https://github.com/jayashreemohan29/Synergy-CoorDL.git
- cd Synergy-CoorDL
- git submodule sync --recursive && git submodule update --init --recursive
- git checkout iterator_chk
- cd docker
- CREATE_RUNNER="YES" ./build.sh
```
 This will create a docker container tagged nvidia/dali:py36_cu10.run

Alternately you could use the docker image hosted [here](https://hub.docker.com/repository/docker/jayashreemohan/synergy_dali) using :
```
docker pull jayashreemohan/synergy_dali:latest
```

### Getting Started
The simplest way to get started with Synergy, is to test it out in a simulated cluster (can be run on a local machine without GPUs, or any specific hardware requirement). The test harness is the [runner.py](simulator/runner.py) file. For instance, to evaluate a FIFO scheduling policy using the default GPU-proportional allocation and synergy's tune based allocation, run the following command:

```
python runner.py --cluster_job_log trace/cluster_job_log --plot  2>&1 | tee  out-deploy
```
Note that, each combination may take upto 5 minutes to complete simulation.

Other options supported by the test harness are:

* --cluster_job_log : The Philly trace
* --plot : Plot the CDF and JCT of runs
* --multigpu : Allow multi-GPU jobs in the mix
* --no_exp :  Disable the exponential arrival distribution
* --philly_arrival : Use arrival information as is from the Philly trace (must also use --no_exp)
* --rec_trace : Record the generated trace
* --replay_trace : Replay a previously recorded trace
* --config_file : Cluster configuration, default of 128GPUs in configs/default_cluster.ini
* --no_simulate : Run it on a real GPU cluster
* schedulers : List of schedulers to run, for eg., ['FIFO+fair' , 'FIFO+tune']
* jobs_per_hour : List of different arrival rates, for eg., np.arange(1.0, 10, 1)
* class split : Split of <vision. language, speech> models, for eg., class_split=[(20,70,10)]


Other detailed run instructions are in [scheduler](simulator/README.md) and [profiler](README-offline-profiler.md)

For more detailed instructions on how to reproduce results from the OSDI paper, see [here](ae-readme.md).
