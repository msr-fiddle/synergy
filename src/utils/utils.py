import atexit
import os

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
		
	
