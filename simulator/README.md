## Synergy Simulation and Deployment

### Defining the cluster config

Create a config file in $ROOT/configs such as  configs/config_file.ini
	-	Contains details of per-machine stats that will be used to track resources
	- Entries could include:
		- [CLUSTER]
		- racks = 1
		- servers_per_rack = 4
		- gpus_per_server = 8
		- cpus_per_server = 24
		- dram_per_server = 500
		- sspeed_per_server = 500

	- Since we assume homogeneous servers, the GPU, CPU IDs to be used for deployment, memory allocation, and scheduling
     decisions will be made based on these metrics.

## Prereq

Tested on Python version 3.6.8 and 3.8
```
- cd deployment
- ./upgrade_pip.sh
- pip install -r requirements.txt
- make
```

### Simulation

In Runner.py : 
	- Uses philly trace with an exponential arrival distribution by default, as in most experiments in the paper
	- Appropriately set the scheduler in the script
 
    - Assumes default_cluster.ini file with 128 GPUs for simulation if no custom config file is specified. If you need a specific cluster config, specify it using --cluster_config option

	- Single-GPU workload trace

		python runner.py --cluster_job_log trace/cluster_job_log --plot  2>&1 | tee  out-deploy


	- Multi-GPU workload trace

		python runner.py --cluster_job_log trace/cluster_job_log --plot --multigpu  2>&1 | tee  out-deploy


### Deployment

- There's one central scheduler server that makes scheduling decisions and launches jobs on appropriate
worker machines in each round
- Each worker machine should run a server to interact with the scheduler and accept requests.


In configs/machine_ip_port.txt:
	- Enter in order (IP, port, start_gpu_id, start_cpu_id, needs_numa_aware_alloc) for each worker machine
  - The # lines (entries)in the above file must be qual to the number of servers mentioned 
      in configs/config_file.ini
	- GPU devices for deployment will be enumerated from (start_gpu_id, gpus_per_server+start_gpu_id)
	- CPU indices for deployment will be enumerated from (start_cpu_id, cpus_per_server+start_cpu_id) if not numa aware
	- Else CPU indices will be retrieved from numactl stats and appropriately allocated


In Runner.py : 

	- Check round duration (set to 300)
	- Uses default_workload in jobs/workload.py and LAS (las_synergy_new)
	- Set job_total_iters and gpu_demands
	- Static trace


	- On the scheduler machine:
		python runner.py --cluster_job_log trace/cluster_job_log --plot --config_file configs/test_deployment.ini --conn_file configs/machine_ip_port.txt  --no_use_cache --no_simulate --num_jobs_default 4 2>&1 | tee  out-deploy

	- On each worker machine:

		python launch_worker_server.py -i 127.0.1.1 -s 14000 -w 16000 -g 8 --run_dir ./ --data_dir ./ --checkpoint_dir ./chk/

		
		python launch_worker_server.py -i 127.0.1.1 -s 14000 -w 16001 -g 8 --run_dir ./ --data_dir ./ --checkpoint_dir ./chk/



Deployment2

python runner.py  --config_file configs/test_deployment.ini --conn_file configs/machine_ip_port.txt  --no_use_cache --no_simulate --num-jobs-default 8 2>&1 | tee out-deploy-synthetic


		python launch_worker_server.py -i 127.0.1.1 -s 14000 -w 16000 -g 8 --run_dir ./ --data_dir ./ --checkpoint_dir ./chk/


Record and replay trace for deployment:

Record trace:
python runner.py --cluster_job_log trace/cluster_job_log --plot  --static --small_static_trace --num-jobs-default 100 --record_trace --no_use_cache --config_file configs/test_deployment.ini  2>&1 | tee static-simulate-fifo-1server-allimage  


Replay trace:
In simulation : python runner.py --plot --static --replay_trace record100.0_fair --no_use_cache --config_file configs/test_deployment.ini  2>&1 | tee  static-simulate-fifo-1server-allimage-replay-simulate

In deployment : python runner.py --no_simulate --plot --static --replay_trace record100.0_fair --conn_file configs/machine_ip_port.txt  --no_use_cache --config_file configs/test_deployment.ini  2>&1 | tee  static-simulate-fifo-1server-allimage-replay-2

python launch_worker_server.py -i 127.0.0.1 -s 14000 -w 16000 -g 8 --run_dir ./ --data_dir ./ --checkpoint_dir ./chk/ 2>&1 | tee out-server-allimg-replay-2



 

