import argparse
import os
import sys
import math
import errno

sys.path.append('./')
sys.path.append('./src')
from utils.utilities import Register
from iterator.synergy_iterator import SynergyIterator

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
    from apex.parallel.LARC import LARC
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

parser = argparse.ArgumentParser(description='Testing Synergy iterator using DALI and PyTorch')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--shuffle_seed", default=0, type=int)
parser.add_argument('--dali', action='store_true')
parser.add_argument('--synergy', action='store_true')
parser.add_argument('--data', metavar='DIR', default="./", type=str,
                    help='path(s) to dataset (if one path is provided, it is assumed\n' +
                    'to have subdirectories named "train" and "val"; alternatively,\n' +
                    'train and val paths can be specified directly by providing both paths as arguments)')

args = parser.parse_args()

def TrainPipe(Pipeline):
	def __init__(self, batch_size, num_threads, device_id, data_dir, resume_index=0, resume_epoch=0):
		super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
		
		if not resume_index and not resume_epoch:
			print("Using baseline DALI"):
			self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, shuffle_after_epoch=True, shuffle_seed=args.shuffle_seed, debug=True)
		else:
			print("Using Synergy-CoorDL"):
			self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, shuffle_after_epoch=True, shuffle_seed=args.shuffle_seed, debug=True, resume_index=resume_index, resume_epoch=resume_epoch)

	def define_graph(self):
		self.jpegs, self.labels = self.input(name="Reader")
		return [self.jpegs, self.lables]


def DALI_loader(resume_index=0, resume_epoch=0):
	pipe = TrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=args.traindir, resume_index=resume_index, resume_epoch=resume_epoch)
	pipe.build()

	if resume_index > 0:
		resume_size = int(pipe.epoch_size("Reader") / args.world_size) - resume_index
		train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size), fill_last_batch=False, resume_size=resume_size)
	else:
		train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size), fill_last_batch=False)

	return pipe, train_loader	

def PyTorch_loader(resume_index=0, resume_epoch=0):
	train_dataset = datasets.ImageFolder( args.traindir)
	train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, sampler=train_sampler)
	return train_loader

def save_shuffle_order(fname, train_pipe=None):
	if args.dali:
		shuffle_order = train_pipe.index_list("Reader")
	else:
		shuffle_order = [(0, 0)]

	with open(fname, 'w+') as sf:
		for idx, ftuple in enumerate(shuffle_order):
			sf.write(str(idx) + ',' + ftuple[0] + '\n')

def load_shuffle_order(fname):
	if not os.path.exists(fname):
		sys.exit('Shuffle order list does not exist')
	with open(fname, 'r') as sf:
		csv_reader = reader(sf)
		shuffle_order = list(map(tuple, csv_reader))
	return shuffle_order

def baseline_chk:
	if args.dali:
		pipe, train_loader = DALI_loader()
	else:
		train_loader = PyTorch_loader()

	# One full epoch
	for epoch in range(0, 1):
		train(train_loader, epoch)

		torch.distributed.barrier()

		if args.dali:
			train_loader.reset()

	# Partial second epoch
	for epoch in range(1, 2):
	
		#Save the index list for comparison later
		if args.local_rank == 0:
			fname = 'chk-epoch-' + str(epoch) + '.log'
			save_shuffle_order(fname)

		samples = train(train_loader, epoch, max_steps=170)
	
		if must_chk and args.local_rank == 0:
			save_checkpoint({
				'epoch': epoch,
				'samples': samples
			}, filename=args.chk)

		torch.distributed.barrier()

		if args.dali:
			train_loader.reset()

	if args.dali:
		del pipe

def baseline_resume()
	if not os.path.exists(args.chk):
		sys.exit('CHK does not exist')
	print("=> loading checkpoint '{}'".format(args.chk))
	checkpoint = torch.load(args.chk)
	start_epoch = checkpoint['epoch']
	# 0 if this epoch is yet to be started
	partial_samples_done = checkpoint['samples']
	
	if args.local_rank == 0:
		fname = 'chk-epoch-' + str(start_epoch) + '.log'
		shuffle_order_old = load_shuffle_order(fname)

	if args.dali:
		pipe, train_loader = DALI_loader()
		shuffle_order_new = pipe.index_list("Reader")

	else:
		train_loader = PyTorch_loader()
		shuffle_order_new = [(0, 0)]

	
	#assert shuffle_order_new == shuffle_order_old

	# Finish the second epoch
	for epoch in range(start_epoch, 2):
		train(train_loader, epoch)
		torch.distributed.barrier()
		if args.dali:
			train_loader.reset()

	if args.dali:
		del pipe


def train(train_loader, epoch, max_steps=-1):
	samples_done = 0
	if args.dali:
		train_loader_len = int(math.ceil(train_loader._size / args.batch_size))
	else:
		train_loader_len =  int(len(train_loader))

	total_samples = train_loader_len*args.batch_size
	
	for i, data in enumerate(train_loader):
		if args.dali:
			images = data[0]["data"]
			target = data[0]["label"]
		else:
			images, target = data
			
		del images
		del target
		samples_done += args.batch_size
		if max_steps >= 0:
			if i  == max_steps - 1:
				return samples_done

	return samples_done % total_samples


def save_checkpoint(state, filename='model.chk'):
	torch.save(state, filename)	


def main():
	print("Test")

	args.chk_dir = './chk/'
	args.chk = args.chk_dir + 'model.chk'
	args.traindir = os.path.join(args.data, 'train')
	args.workers = 3
	args.batch_size = 512
	#args.world_size = 1
	args.world_size = torch.distributed.get_world_size()	


	#create chk dir
	try:
		os.makedirs(args.chk_dir)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise


	

if __name__ == "__main__":
	main()
