import sys
import os
import time
import traceback
import argparse

#from runtime.rpc import scheduler_client, scheduler_server
from synergy_iterator import SynergyIterator
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torchvision.models as models
import torchvision.transforms as transforms
from multiprocessing import Process
#Dummy scheduler code which registers workers based on cluster config

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training using DALI')
parser.add_argument('--data', metavar='DIR', default="./", type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,  
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('--job-id',  default=0, type=int)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                   help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=3, type=int, metavar='N',
                   help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                   help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                   metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--gpus',  default=1, type=int,
                   metavar='N', help='Num GPUs (default: 1)')
parser.add_argument('-i', "--max-iterations", default=-1, type=int)
parser.add_argument("--sched-addr", default='127.0.0.1', type=str)
parser.add_argument("--sched-port", default='14000', type=str)
parser.add_argument("--log-dir", default='./', type=str)


args = parser.parse_args()
print(args)


def main():
    dummy_dl = list(range(args.max_iterations))
    synergy_iterators = [SynergyIterator(dummy_dl, mock=True) for _ in range(args.gpus)]
    #TODO : Create seperate iterators processes
    procs = []
   

    for gpu in range(args.gpus):
        p = Process(target=train, args=(synergy_iterators[gpu], len(dummy_dl), args.workers,))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


def train(synergy_iterator, length, cpus):
    for i,_ in enumerate(synergy_iterator):
        #for j in range(1, args.gpus):
        #next(synergy_iterator)
        if cpus == 3:
            time.sleep(0.05) 
        else:
            sleep = 3*0.05/cpus
            time.sleep(sleep) 
 
        print("Processing iteration {}/{} : R {} : Steps : {}/{}".format(i+1, length, synergy_iterator._round, synergy_iterator._steps_this_round, synergy_iterator._num_steps_per_round))
        print("Processing iteration {}/{} : R {} : Steps : {}/{}".format(i+1, length, synergy_iterator._round, synergy_iterator._steps_this_round, synergy_iterator._num_steps_per_round))
        print("Processing iteration {}/{} : R {} : Steps : {}/{}".format(i+1, length, synergy_iterator._round, synergy_iterator._steps_this_round, synergy_iterator._num_steps_per_round))

    #for iterator in synergy_iterators: 
    synergy_iterator.complete()
    print("Job complete!")
    


if __name__ == "__main__":

    os.environ["SYNERGY_JOB_ID"] = str(args.job_id)    
    os.environ["SYNERGY_SCHED_ADDR"] = args.sched_addr
    os.environ["SYNERGY_SCHED_PORT"] = args.sched_port 
    os.environ["SYNERGY_LOG_DIR"] = args.log_dir 
    os.environ["SYNERGY_DEBUG"] = "true" 
   
    main()   

