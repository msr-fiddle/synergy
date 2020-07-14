"""
An offline profiler that varies CPU and memory and records job performance

Input :  1. Job script
				 2. Number of GPUs
				 3. Docker container image
				 4. Assumes that the following parameters are exposed:
						a. Num iterations to stop at ( --max_iterations)
						b. Data path ( --data)
						c. Num parallel data workers ( --workers)

Output :	
					Performance_profile_<job_name>_<timestamp>.json
						1. Empirical perf with varying CPU
						2. Predicted perf with varying memory
"""


import sys
import subprocess
import os
from argparse import ArgumentParser, REMAINDER 
import multiprocessing  
import json   
import time
from datetime import datetime

sys.path.append("../")
from src.utils import utils

def parse_args():
	parser = ArgumentParser(description="Offline performance profiler")

	# These arguments are passed by the scheduler, from the job submission
	# file. Training scripts are always single-GPU scripts, and distributed
	# training is handled by torch.distributed.launch

	# The master IP, port, num_nodes and node_ranks are got from the 
	# placement module of the scheduler

	# The user inputs docker-img, num-gpus, training script and its args
 	
	parser.add_argument("--docker-img", type=str, default=None,
														help="Docker image to use for the job")
	parser.add_argument("--num-gpus", type=int, default=1,
														help="Number of GPUs req by the job")
	parser.add_argument("--job-name", type=str, default="job-0",
														help="Unique named identifier for the job")
	parser.add_argument("--nnodes", type=int, default=1,       
														help="The number of nodes to use for distributed "  
														"training")
	parser.add_argument("--node_rank", type=int, default=0, 
														help="The rank of the node for multi-node" 
														"distributed training")  
	parser.add_argument("--nproc_per_node", type=int, default=1,
													help="The number of processes to launch on each node, "
														"for GPU training, this is recommended to be set "
														"to the number of GPUs in your system so that "
														"each process can be bound to a single GPU.")
	parser.add_argument("--master_addr", default="127.0.0.1", type=str,
													help="Master node (rank 0)'s address, should be either"
														"the IP address or the hostname of node 0, for "
														"single node multi-proc training, the "
														"--master_addr can simply be 127.0.0.1")
	parser.add_argument("--master_port", default=29500, type=int,
													help="Master node (rank 0)'s free port that needs to "
														"be used for communciation during distributed "
														"training")
	# Cluster specific
	parser.add_argument("--cpu", type=int, default=0,
													help="Fix max number of CPUs to profile")
	parser.add_argument("--memory", type=int, default=0,
													help="Fix max memory to profile")


	# The following arguments are expected to be supported by the
	# training script. Max iterations should supersede epochs.
	# It is set to -1 by default in the script to ignore iter count
	parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
													help='number of data loading workers')
	parser.add_argument('--epochs', default=1, type=int, metavar='N',
													help='number of total epochs to run')
	parser.add_argument("--max-iterations", default=50, type=int,
													help='max number of minibatches to profile')
	parser.add_argument("--data", default="./", type=str,
													help='Path to dataset (blob mnt or local')


	# Profiler specific
	parser.add_argument('--profiler-src', default="/mnt2/jaya/store-aware-packing/src/", type=str,
													help='Path to the profiler src')
	parser.add_argument('--container-mnt', default="/datadrive/", type=str,
													help='Mountpoint in the cointainer where src code is present')


	# The user given training script and any optional args to the script
	parser.add_argument("training_script", type=str, 
													help="The full path to the single GPU training " 
														"program/script to be launched in parallel, "
														"followed by all the arguments for the "
														"training script") 
	parser.add_argument('training_script_args', nargs=REMAINDER)

	return parser.parse_args()

args = parse_args()

def print_inputs():
	print("--------- Offline Sensitivty Profiler ----------")
	print("Job name = {}".format(args.job_name))
	print("GPUs = {}".format(args.num_gpus))
	print("Docker image = {}".format(args.docker_img))
	print("Workers = {}".format(args.workers))
	print("Max iterations = {}".format(args.max_iterations))
	print("Training script = {}".format(args.training_script))
	print("Training script args = {}".format(args.training_script_args))
	print("Profile path = {}".format(args.profile_path))


def withinTenPercent(orig, current):
	orig = float(orig)
	current = float(current)
	if orig <= 0:
		return True

	if abs((orig - current)/orig*100) < 10:
		print("Within 10% of {} and {} is True".format(orig, current))
		return True
	print("Within 10% of {} and {} is False".format(orig, current))
	return False 

def launch_job(job_name, container, cpu, memory, script, script_args, out_dir):
	monitor = utils.Monitor(name = job_name)
	cmd = "nvidia-docker run " + \
				" --ipc={}".format("host") + \
				" --mount src={},target={},type={}".format("/", args.container_mnt, \
				"bind")+\
				" -it" + \
				" --rm" + \
				" --network={}".format("host") + \
				" --cpus=\"{}\"".format(cpu) + \
				" --cpuset-cpus={}-{}".format(0, cpu-1) + \
				" --memory={}g".format(memory) + \
				" --privileged" + \
				" -e \"OUT_DIR=\"{}\"\"".format(out_dir) + \
				" " + container + \
				" /bin/bash -c " + \
				" \'cd {}; ".format(args.container_mnt + args.profiler_src) + \
				" export PYTHONPATH=\"$PYTHONPATH:{}\";".format(args.container_mnt + args.profiler_src) + \
				" echo $PYTHONPATH; " + \
				" python -m profiler.launch" + \
				" --nproc_per_node={}".format(args.num_gpus) + \
				" " + script + \
				" " + ' '.join(map(str, script_args)) + \
				" --workers={}".format(int(cpu/args.num_gpus)) + \
				" --job-name={}".format(job_name) + "\'"

	print(cmd)
	process = subprocess.Popen(cmd, env = os.environ.copy(), shell=True)
	process.wait()
	res = monitor.peekLog()
	return res[0]



def profile_cpu(num_cpus=0):
	actual_num_cpus = multiprocessing.cpu_count()
	if num_cpus <= 0 or num_cpus > actual_num_cpus:
		num_cpus = actual_num_cpus
	print(num_cpus)
	 
	res = -1
	cpu_to_profile = num_cpus
	while True:
		start = time.time()
		job_name = args.job_name + "_cpu_" + str(cpu_to_profile)
		cur_res = launch_job(job_name, args.docker_img, cpu_to_profile, 500, args.training_script, args.training_script_args, "./")
		#break
		if withinTenPercent(res, cur_res):
			cpu_to_profile = cpu_to_profile/2
			res = cur_res
		else:
			cpu_to_profile = cpu_to_profile - 1
			res = cur_res
		
		dur = time.time() - start
		print("Time to explore {} = {}s".format(job_name, dur))
		if cpu_to_profile < 1:
			break
		
		

def main():
	
	# Create output json profile
	dateTimeObj = datetime.now()
	timestampStr = dateTimeObj.strftime("%d-%b-%Y-(%H:%M:%S.%f)")
	args.profile_dir = os.getcwd() + "/profiles/" + args.job_name +  "_" + timestampStr + "/" 
	args.profile_path = args.profile_dir + "profile.json"
	try:
		os.makedirs(args.profile_dir)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

	print_inputs()
	
	profile_cpu(num_cpus=args.cpu)

if __name__ == "__main__":
	main()
