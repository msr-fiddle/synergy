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
parser.add_argument('-i', "--max-iterations", default=-1, type=int)
parser.add_argument("--sched-addr", default='127.0.0.1', type=str)
parser.add_argument("--sched-port", default='14000', type=str)


args = parser.parse_args()
print(args)


def main():
    dummy_dl = list(range(args.max_iterations))
    synergy_iterator = SynergyIterator(dummy_dl, mock=True)
    for i,_ in enumerate(synergy_iterator):
        """
        Assume iter durations are 300 ms each
        """
        #time.sleep(0.3)
        print("Processing iteration {}/{} : R {} : Steps : {}/{}".format(i+1, len(dummy_dl), synergy_iterator._round, synergy_iterator._steps_this_round, synergy_iterator._num_steps_per_round)) 
    synergy_iterator.complete()
    print("Job complete!")
    


if __name__ == "__main__":
    #args = sys.argv
    #if len(args) != 4:
    #    print("Usage python job_script.py [sched ip] [sched port] [job_id] []")
    #    sys.exit(1)

    os.environ["SYNERGY_JOB_ID"] = str(args.job_id)    
    os.environ["SYNERGY_WORKER_ID"] = "234" 
    os.environ["SYNERGY_SCHED_ADDR"] = args.sched_addr
    os.environ["SYNERGY_SCHED_PORT"] = args.sched_port 
    os.environ["SYNERGY_LOG_DIR"] = "." 
    os.environ["SYNERGY_DEBUG"] = "true" 
   
    main()   

