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

To run various combinatons of [schedulers](simulator/runner.py#L393), [class_split](simulator/runner.py#L408), and load [jobs_per_hours](simulator/runner.py#L405), please refer to these respective lines in `runner.py`.

The permissible values for schedulers are 'LAS', 'SRTF', 'FIFO', 'FTF' - these options run the Synergy-Greedy version. To enable fair version, use 'scheduler+fair', and for Synergy-Tune, use 'scheduler+tune'. For example, 'LAS+fair', 'LAS+tune'.

### Testing the functionality

The quickest way to check if the simulator functions correctly is to execute the runner script with default setting 
```
python runner.py --cluster_job_log trace/cluster_job_log --plot  2>&1 | tee  out.log
```
This script runs a FIFO synergy tune scheduler for a load of 9 jobs/hr using the philly-derived trace for a class split of (20,70,10)

The detailed log containing each jobs arrival and finish times can be found in `out.log`. Additionally, at the end of the run, a summary of mean and 99th percentile JCT for the jobs is presented as follows :
```
Running scheduler FIFO+tune - ('FIFO-Tune', 9.0, (20, 70, 10))
Mean =  21.876600555555555
99th =  156.93307777777775
99.9th =  170.61199750000023
```

As a quick point of validation, you can compare these numbers to those presented in Fig 8, for Synergy-Tune. 
After each run, the directry `plots`contains the corresponding plots of `avg JCT vs load` and `cdf for each load`, which have been presented throughout 
the evaluation in the paper. The `cache` directory contains the results from the run for each configuration, should you rerun the experiment to plot different combinations of schedulers, stats from previously cached runs will be used. Use `--no_use_cache`if you want to force rerun the config, and `--no_cache_result' to disable caching. 

### Fig 1 (Average JCT with Synergy)
To reproduce the intro result (Fig 1), uncomment the appropriate [scheduler](simulator/runner.py#L397) and [load](simulator/runner.py#L406) arguments in `runner.py` and execute the runner in the same way as above with an additional `--multigpu` option. The resultant graph is in `plots/avg_jct_vs_load.png`

### Philly trace evaluation
By default, the simulation uses a philly-derived trace as explained in the paper. To evaluate a larger cluster with the arrival times and gpu load as in the
actual philly trace (Fig 6). The load parameter is immaterial in this mode. Set the scheduler parammeter to the types you want to test.

```
python runner.py --cluster_job_log  trace/cluster_job_log --plot --philly_arrival --no_exp --multigpu --config_file configs/philly_cluster.ini  2>&1 | tee  out-philly.log
```


### Philly-derived trace evaluation
To reproduce Fig 7 and 8, execute the runner as usual, with (--multigpu for Fig 7), by setting appropriate values for scheduler and load.

For single-GPU jobs (Fig 8), 
```
python runner.py --cluster_job_log trace/cluster_job_log --plot  2>&1 | tee  out.log
```

For multi-GPU jobs (Fig 7),
```
python runner.py --cluster_job_log trace/cluster_job_log --plot --multigpu 2>&1 | tee  out.log
```

For Fig 9, uncomment the list of splits [here](simulator/runner.py#L409) and execute the runner as usual.

To vary CPU:GPU config, use the appropriate config file as input to the runner, and rename the model directory appropriately. For instance, to 
test a CPU:GPU ratio of 4, 

```
mv models models-3
mv models-4 models
python runner.py --cluster_job_log trace/cluster_job_log --plot --config_file configs/cluster_4.ini  2>&1 | tee  out.log
```

For Fig 12, please set schedulers to [this](simulator/runner.py#L400).

### Deployment traces

To run the deployment experiments, both the statc and dynamic traces used in the paper are available under `deploy_traces`. Following deployment instructons [here](simulator/README.md#deployment), run the scheduler with 
```
'--replay_trace deploy_trace/static_100job_6_3_1100.0 --static' for static trace
'--replay_trace deploy_trace/dynamic/srtf.log' for dynamic trace
```
