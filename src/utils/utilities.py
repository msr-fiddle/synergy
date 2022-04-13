import atexit
import os
from collections import OrderedDict
import sys

class Register(object):
	def __init__(self, name="job-0"):
		self.name = name 
		self.env_batch = "AVG_BATCH_" + name
		self.env_iter = "ITER_" + name
		self.env_epoch = "EPOCH_" + name
		self.file = "PERF_" + name + ".txt"

		atexit.register(self.save_perf)

	def save_perf(self):
		with open(self.file, 'w+') as sp:
			if self.env_batch in os.environ:
				sp.write(os.environ[self.env_batch] + "\n")
			if self.env_iter in os.environ:
				sp.write(os.environ[self.env_iter] + "\n")
			if self.env_epoch in os.environ:
				sp.write(os.environ[self.env_epoch] + "\n")

	def log_update(self, batch_time, iteration, epoch):
		os.environ[self.env_batch] = str(batch_time)
		os.environ[self.env_iter] = str(iteration)
		os.environ[self.env_epoch] = str(epoch)


class Monitor(object): 
	def __init__(self, name="job-0"):
		self.name = name 
		self.file = "PERF_" + name + ".txt"

	def peekLog(self):
		if not os.path.exists(self.file):
			return [-1,-1,-1]
		with open(self.file, 'r') as rf:
			val = rf.readlines()
			if len(val) > 3:
				return val[:3]
			else:
				return val
		
'''
Access the performance of a job for 'c' CPU and 'm' memory GB
perf =  PerfMatrix()
perf.put(c, m, val)

val = perf.get(c, m)

best_c, best_m = perf.getBest(stop=10) #ignore changes within 10%

perf.persist(path="path_to_json") # Writes out the matrix to disk
'''

class PerfMatrix:
	def __init__(self, path="job-0.json"):
		self.perf_dict = OrderedDict()
		self.file = path

	def put(self, cpu, mem, val):
		if cpu not in self.perf_dict:
			self.perf_dict[cpu] = OrderedDict()
		
		self.perf_dict[cpu][mem] = val
		return True
				
	def get(self, cpu, mem):
		if cpu not in self.perf_dict:
			return False
		else:
			if mem not in self.perf_dict[cpu]:
				return False
			else:
				return self.perf_dict[cpu][mem]		

	def show(self):
		cpu_keys = self.perf_dict.keys()
		cpu_keys = sorted(cpu_keys)
		#cpu_keys.sort()
		prev_mem_keys = None
		for cpu in cpu_keys:
			mem_keys = self.perf_dict[cpu].keys()
			if prev_mem_keys is None:
				prev_mem_keys = mem_keys
				continue
			if len(list(set(mem_keys) & set(prev_mem_keys))) != len(mem_keys):
				print("Not a matrix")
				
		prev_mem_keys = sorted(prev_mem_keys)
		#prev_mem_keys.sort()	
		sys.stdout.write("\n  \t")
		for mem in prev_mem_keys:
			sys.stdout.write("{}\t".format(mem))
		sys.stdout.write("\n")

		for cpu in cpu_keys:
			sys.stdout.write("{}\t".format(cpu))
			for mem in prev_mem_keys:
				val = self.perf_dict[cpu][mem]
				sys.stdout.write("{:.0f}\t".format(val))    
				#print("{} : {}GB = {}".format(cpu, mem, val))
			sys.stdout.write("\n")

	def write(self):
		if not os.path.exists(self.file):
			with open(self.file, 'w') as fp:
				json.dump(self.perf_dict, fp)
