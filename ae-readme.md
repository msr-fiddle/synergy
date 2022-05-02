This document outlines how to run the main experiments in he OSDI'22 paper

## Setup

To run all the simulation experiments, please install dependencies as enumerated below.
```
- cd simulator/deployment
- ./upgrade_pip.sh
- make
- cd ..
- pip install -r requirements.txt
```

Additionally, to setup the profiler, 
```
- cd src
- cd profiler; ./prereq.sh; cd ..
```

For the physical cluster experiments, build synergy iterator in a docker container as explained [here](README.md#setup-iterator).

## Reproducing results

We have enumerated the required arguments to reproduce the key results in our paper. The evaluation in the paper largely shows results on a simulated cluster. 
This can be run on a local machine with no GPUs. We have profiled the throughputs of different models in the experimental setup as described in Section 5.1 in 
the paper, and these profiles can be found in `simulator/models/`. Execute all the following simulation runs from the `simulator` sub-directory using the file 
`runner.py`. By default, the cluster configuration is 128 V100 GPUs (16 * 8-GPU machines, each with 500GB DRAM, and 24 CPU cores).

### Fig 1 (Average JCT with Synergy)


### Philly trace evaluation


### Philly-derived trace evaluation
