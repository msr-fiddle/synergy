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
import datetime

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

	# The following arguments are expected to be supported by the
	# training script. Max iterations should supersede epochs.
	# It is set to -1 by default in the script to ignore iter count
	parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
													help='number of data loading workers')
	parser.add_argument('--epochs', default=1, type=int, metavar='N',
													help='number of total epochs to run')
	parser.add_argument("--max-iterations", default=50, type=int,
													help='max number of minibatches to profile')


	# The user given training script and any optional args to the script
	parser.add_argument("training_script", type=str, 
													help="The full path to the single GPU training " 
														"program/script to be launched in parallel, "
														"followed by all the arguments for the "
														"training script") 
	parser.add_argument('training_script_args', nargs=REMAINDER)

	return parser.parse_args()

args = parse_args()


def main():
	print_header()
	
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

	


if __name__ == "__main__":
	main()
