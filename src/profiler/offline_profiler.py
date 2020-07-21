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
#import numpy
from collections import OrderedDict 

sys.path.append("../")
sys.path.append("./")
from utils.utilities import PerfMatrix
from utils.utilities import Monitor

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
	parser.add_argument("--memory", type=int, default=500,
													help="Fix max memory to profile")


	# The following arguments are expected to be supported by the
	# training script. Max iterations should supersede epochs.
	# It is set to -1 by default in the script to ignore iter count
	parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
													help='number of data loading workers')
	parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
													help='Per-gpu batch size')
	parser.add_argument('--epochs', default=1, type=int, metavar='N',
													help='number of total epochs to run')
	parser.add_argument("--max-iterations", default=50, type=int,
													help='max number of minibatches to profile')
	parser.add_argument("--data", default="./", type=str,
													help='Path to dataset (blob mnt or local')
	parser.add_argument("--str-bw", default="./STR_BW", type=str,
													help='Path to dataset (blob mnt or local')
	parser.add_argument("--mem-thr", default="./MEM_THR", type=str,
													help='Path to dataset (blob mnt or local')


	# Profiler specific
	parser.add_argument('--profiler-src', default="/mnt2/jaya/store-aware-packing/src/", type=str,
													help='Path to the profiler src')
	parser.add_argument('--container-mnt', default="/datadrive/", type=str,
													help='Mountpoint in the cointainer where src code is present')
	parser.add_argument('--local-disk', default="/mnt2", type=str,
													help='Local disk mount point')

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

def get_data_path(arg_list):
	if '--data' not in arg_list:
		return arg_list[-1]
	else:
		i = arg_list.index('--data')
		return arg_list[i+1]


def get_dataset_stats(dir_path):
	print("Dir path = {}".format(dir_path))
	train_path = dir_path + "/train/"
	cmd = "du -sh " + train_path
	process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
	(output,err)=process.communicate()
	exit_code = process.wait()
	if exit_code != 0:
		return 0, 1889601
	
	size = output.decode('utf-8').split()[0][:-1]

	cmd = "find " + train_path + " -type f | wc -l"
	process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
	(output,err)=process.communicate()
	exit_code = process.wait()
	samples = output.decode('utf-8').split()[0]

	print("Size = {}, samples ={}".format(size, samples))
	return int(size), int(samples)



def launch_job(job_name, container, cpu, memory, script, script_args, out_dir):
	monitor = Monitor(name = job_name)
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
				" -b {}".format(args.batch_size) + \
				" --job-name={}".format(job_name) + "\'"

	print(cmd)
	process = subprocess.Popen(cmd, env = os.environ.copy(), shell=True)
	process.wait()
	res = monitor.peekLog()
	return float(res[0].rstrip())

def launch_job_dummy(cpu):
	#per epoch time
	return 2067*pow(cpu, -0.649)


def launch_job_dummy_oi(cpu):
	if cpu == 24:
		return 0.804
	elif cpu == 12:
		return 1.41
	elif cpu == 8:
		return 1.53
	elif cpu == 4:
		return 2.3
	else: 
		return 0
	#return 2967*pow(cpu, -0.649)

def profile_cpu(num_cpus=0):
	actual_num_cpus = multiprocessing.cpu_count()
	if num_cpus <= 0 or num_cpus > actual_num_cpus:
		num_cpus = actual_num_cpus
	print(num_cpus)
	 
	cpu_profile_points = OrderedDict()
	
	# TODO : Get dataset stats 
	num_samples = args.num_samples
	#num_samples = 1281167
	num_iters = int(num_samples/args.num_gpus/args.batch_size)
	res = -1
	cpu_to_profile = num_cpus
	while True:
		start = time.time()
		job_name = args.job_name + "_cpu_" + str(cpu_to_profile)
		#cur_res = launch_job(job_name, args.docker_img, cpu_to_profile, 500, args.training_script, args.training_script_args, "./") 
		cur_res = launch_job_dummy_oi(cpu_to_profile)
		cur_res = cur_res * num_iters
		print("Epoch time = {}, num_iters={}".format(cur_res, num_iters))
		cpu_profile_points[cpu_to_profile] = cur_res
		#break
		if withinTenPercent(res, cur_res):
			cpu_to_profile = cpu_to_profile/2
			res = cur_res
		else:
			cpu_to_profile = cpu_to_profile - args.num_gpus
			res = cur_res
		
		dur = time.time() - start
		print("Time to explore {} = {}s".format(job_name, dur))
		if cpu_to_profile < 1:
			break

	return cpu_profile_points


def get_bandwidths(disk=args.local_disk, cpu=1):
	str_bw = get_storage_bandwidth(disk=disk)
	mem_bw = get_mem_throughput(cpu=cpu)
	return mem_bw, str_bw

def memProfileExists(cpu):
	mem_str = 'MEM_THR' + '_' + str(cpu)
	mem_file = args.mem_thr + '_' + str(cpu)
	if mem_str in  os.environ:	
		mem_thr = os.environ[mem_str]
		return float(mem_thr)
	elif os.path.exists(mem_file):
		with open(mem_file, 'r') as rf:
			mem_thr = rf.readline()
		return float(mem_thr)
	else:
		return None
	


def get_mem_throughput(cpu=1):
	mem_str = 'MEM_THR' + '_' + str(cpu)
	mem_file = args.mem_thr + '_' + str(cpu)
	mem_thr = memProfileExists(cpu)
	if mem_thr is not None:
		return mem_thr
	else:
		cmd = "./profiler/memtest " + str(cpu)
		process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True) 
		(output,err)=process.communicate()  
		#print(output.decode('utf-8')) 
		mem_thr = process.wait()   
		#print("GB/s = {}".format(exit_code))
		os.environ[mem_str] = str(mem_thr)
		with open(mem_file, 'w+') as wf:
			wf.write(str(mem_thr))
		return mem_thr


def strProfileExists():
	if 'STR_BW' in  os.environ:
		str_bw = os.environ['STR_BW']
		return float(str_bw)
	elif os.path.exists(args.str_bw):
		with open(args.str_bw, 'r') as rf:
			str_bw = rf.readline()
		return float(str_bw)
	else:
		return None

	
def get_storage_bandwidth(disk=args.local_disk):
	str_bw = strProfileExists()
	if str_bw is not None:
		return str_bw
	else:
		#dev_cmd = "grep \"" + disk + "\" /proc/mounts | cut -d ' ' -f 1"	
		dev_cmd = ['grep', disk, '/proc/mounts']
		dev_cmd_cut = ['cut', '-d', ' ', '-f', '1']
		#print(dev_cmd)
		p = subprocess.Popen(dev_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		#out, err = p.communicate()
		output = subprocess.check_output(dev_cmd_cut, stdin=p.stdout)
		p.wait()
		if p.returncode != 0: 
			print("Error : {}".format(err.decode('utf-8')))
			return 0,0
		device = output.decode('utf-8').rstrip()
		#print(p.stdout.decode('utf-8'), p.stderr.decode('utf-8'))
		#print(out.decode('utf-8'), err.decode('utf-8'))
		#print(p.stdout.decode('utf-8'))
		print("Measuring bandwidth of storage dev  {}".format(device))
		dev_bw = ['sudo', 'hdparm', '-t', device]
		p = subprocess.Popen(dev_bw, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = p.communicate()
		result = out.decode('utf-8')
		print(result, err.decode('utf-8'))
		str_bw = result.split()[-2]
		os.environ['STR_BW'] = str_bw
		with open(args.str_bw, 'w+') as wf:
			wf.write(str_bw)
		return str_bw

def estimate_speed(cpu, mem, max_time):
	# img_size Get avg image size
	#
	disk_bw = strProfileExists() #Get it from file or envt
	mem_thr = memProfileExists(cpu) # Get from file
	dataset_size  = 555
	#dataset_size  = 140
	total_samples = 1889601
	#total_samples = 1281167
	cache_size = float(mem)/100.0*dataset_size 
	disk_fetch_size = dataset_size - cache_size
	time_to_cache = cache_size/mem_thr
	time_to_disk = disk_fetch_size*1024/disk_bw #assume MB/s disk_bw
	#print("{}:{}:{}:{}GB:{}GBcache".format(cpu, mem, time_to_disk, disk_fetch_size, cache_size))
	total_time = time_to_cache + time_to_disk
	avg_sample_size = dataset_size*1024*1024 / total_samples
	effective_store_thr = dataset_size*1024*1024/total_time/avg_sample_size

	if total_time > max_time:
		return total_time
	else:
		return max_time
	

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

	data_path = get_data_path(args.training_script_args)
	
	args.dataset_size, args.num_samples = get_dataset_stats(data_path)
	print("Path = {}, Size = {}, Num samples = {}".format(data_path, args.dataset_size, args.num_samples))

	perf_matrix =  PerfMatrix(path=args.profile_path)
	
	cpu_profile = profile_cpu(num_cpus=args.cpu)

	print(cpu_profile)

	cpu_profile_points = cpu_profile.keys()
	sorted(cpu_profile_points)
	#cpu_profile_points.sort()
	mem_profile_points = [i for i in range(0, 101, 10)]
	num_profile_points_cpu = len(cpu_profile.keys())
	num_profile_points_mem = len(mem_profile_points)
	print(mem_profile_points)
	print(cpu_profile_points)
	#profile_matrix =  numpy.zeros(num_profile_points_cpu, num_profile_points_mem)

	str_bw = get_storage_bandwidth()
	mem_thr_list = []
	for cpu in cpu_profile_points:
		mem_thr_list.append(get_mem_throughput(cpu))
	print("Storage bandwidth = {} MB/s".format(str_bw))
	print(mem_thr_list)
	#print("Mem throughput = {} GB/s".format(mem_thr))

	for key, val in cpu_profile.items():
		perf_matrix.put(key, 100, val)

	#for cpu  in [24]:
	for cpu  in cpu_profile_points:
		for mem in mem_profile_points:
			if mem == 100:
				continue
			time = estimate_speed(cpu, mem, perf_matrix.get(cpu, 100))
			#print("Time for {}:{}% = {}".format(cpu, mem,time))
			perf_matrix.put(cpu, mem, time)		

	perf_matrix.show()

if __name__ == "__main__":
	main()
