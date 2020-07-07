import argparse
import os
import shutil
import time
import math

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torch.nn.functional as F
import subprocess
import sys
import multiprocessing
import atexit
from signal import SIGKILL, SIGTERM
from os import kill, getpid
import audio_model
from audio_iterator.pytorch import DALIAudioClassificationIterator
import torch.multiprocessing as mp 
from torch.multiprocessing import Pool, Process, set_start_method 
try: 
    set_start_method('spawn') 
except RuntimeError:
    pass


try:
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
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training using DALI')
parser.add_argument('data', metavar='DIR', nargs='*',
                    help='path(s) to dataset (if one path is provided, it is assumed\n' +
                    'to have subdirectories named "train" and "val"; alternatively,\n' +
                    'train and val paths can be specified directly by providing both paths as arguments)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=3, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--classes', default=1000, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--fp16', action='store_true',
                    help='Run model fp16 mode.')
parser.add_argument('--dali_cpu', action='store_true',
                    help='Runs CPU based version of DALI pipeline.')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                    '--static-loss-scale.')
parser.add_argument('--prof', dest='prof', action='store_true',
                    help='Only run 10 iterations for profiling.')
parser.add_argument('-t', '--test', action='store_true',
                    help='Launch test mode with preset arguments')

parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--cache_size", default=0, type=int)
parser.add_argument('--sync_bn', action='store_true',
        help='enabling apex sync BN.')
parser.add_argument('--opt-level', type=str)
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)
parser.add_argument('--channels-last', type=bool, default=False)
parser.add_argument('--deterministic', action='store_true')
parser.add_argument('--dist', action='store_true')
parser.add_argument('--unified', action='store_true')
parser.add_argument('--noeval', action='store_true')
parser.add_argument('--pin', action='store_true')
parser.add_argument('--amp',action='store_true',help='Run model AMP (automatic mixed precision) mode.')


#Launch helper
# Optional arguments for the launch helper
parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
parser.add_argument("--node_rank", type=int, default=0,
                        help="The rank of the node for multi-node distributed "
                             "training")
parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")
parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training")
parser.add_argument("--use_env", default=False, action="store_true",
                        help="Use environment variable to pass "
                             "'local rank'. For legacy reasons, the default value is False. "
                             "If set to True, the script will not pass "
                             "--local_rank as argument, and will instead set LOCAL_RANK.")

parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument("--num_jobs", type=int, default=1,
                        help="The number of jobs to run "
                             "for HP search")
parser.add_argument("--num_gpu_per_job", type=int, default=1,
                        help="The number of GPUs to use per "
                             "HP search job")




cudnn.benchmark = True

compute_time_list = []
data_time_list = []



class AudioTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, dali_cpu=True, args_new=None, shuffle_seed=0):
        super(AudioTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        #self.input = ops.FileReader(file_root=data_dir, shard_id=args_new.local_rank_per_job, num_shards=args_new.world_size, shuffle_after_epoch=True, cache_size=5500, shuffle_seed=shuffle_seed)
        #self.input = ops.FileReader(file_root=data_dir, shard_id=args_new.local_rank_per_job, num_shards=args_new.world_size, random_shuffle=True, shuffle_seed=shuffle_seed)
        self.input = ops.FileReader(file_root=data_dir, shard_id=args_new.local_rank_per_job, num_shards=args_new.world_size, shuffle_after_epoch=True, shuffle_seed=shuffle_seed)
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        self.decode = ops.AudioDecoder(device="cpu", dtype=types.FLOAT, downmix=True, sample_rate=8192, downsample_size=160000)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        self.audio, self.label = self.input(name="Reader")
        dec_audio, rate = self.decode(self.audio)
        return [dec_audio, self.label, rate]


class AudioUnifiedTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, dali_cpu=True, args_new=None):
        super(AudioUnifiedTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        #self.input = ops.FileReader(file_root=data_dir, shard_id=args_new.local_rank, num_shards=args_new.num_procs, shuffle_after_epoch=True, cache_size=5500)
        #self.input = ops.FileReader(file_root=data_dir, shard_id=args_new.local_rank, num_shards=args_new.num_procs, random_shuffle=True)
        self.input = ops.FileReader(file_root=data_dir, shard_id=args_new.local_rank, num_shards=args_new.num_procs, shuffle_after_epoch=True)
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        self.decode = ops.AudioDecoder(device="cpu", dtype=types.FLOAT, downmix=True, sample_rate=8192, downsample_size=160000)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        self.audio, self.label = self.input(name="Reader")
        dec_audio, rate = self.decode(self.audio)
        return [dec_audio, self.label, rate]


class AudioValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, dali_cpu=True, args_new=None, shuffle_seed=0):
        super(AudioValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        #self.input = ops.FileReader(file_root=data_dir, shard_id=args_new.local_rank_per_job, num_shards=args_new.world_size, random_shuffle=False,, shuffle_seed=shuffle_seed)
        self.input = ops.FileReader(file_root=data_dir, shard_id=args_new.local_rank_per_job, num_shards=args_new.world_size, random_shuffle=False, shuffle_seed=shuffle_seed)
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        self.decode = ops.AudioDecoder(device="cpu", dtype=types.FLOAT, downmix=True, sample_rate=8192, downsample_size=160000)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        self.audio, self.label = self.input(name="Reader")
        dec_audio, rate = self.decode(self.audio)
        return [dec_audio, self.label, rate]


class AudioUnifiedValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, dali_cpu=True, args_new=None):
        super(AudioUnifiedValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=args_new.local_rank, num_shards=args_new.num_procs, random_shuffle=False)
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        self.decode = ops.AudioDecoder(device="cpu", dtype=types.FLOAT, downmix=True, sample_rate=8192, downsample_size=160000)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        self.audio, self.label = self.input(name="Reader")
        dec_audio, rate = self.decode(self.audio)
        return [dec_audio, self.label, rate]




best_prec1 = 0
args = parser.parse_args()

if args.unified:
    try:
        from nvidia.dali.plugin.pytorch_neuralcache import DALIUnifiedAudioClassificationIterator
    except ImportError:
        raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

# test mode, use default args for sanity test
if args.test:
    args.fp16 = False
    args.epochs = 1
    args.start_epoch = 0
    args.arch = 'resnet50'
    args.batch_size = 64
    args.data = []
    args.prof = True
    args.data.append('/data/imagenet/train-jpeg/')
    args.data.append('/data/imagenet/val-jpeg/')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.local_rank)
    torch.set_printoptions(precision=10)

if not len(args.data):
    raise Exception("error: too few arguments")

if args.amp:
    args.opt_level='O1'


#print("Using mixed precision : {}".format(args.amp))
#print("opt_level = {}".format(args.opt_level))
#print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
#print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))


if args.gpu is not None: 
    warnings.warn('You have chosen a specific GPU. This will completely '    
                  'disable data parallelism.')  

total_gpu = torch.cuda.device_count() 
args.num_procs = args.num_jobs * args.num_gpu_per_job
if args.num_procs > total_gpu:
    raise Exception("error: Cannot run HP - more GPU requested than available") 

args.batch_maps = []
args.counter_maps = []
args.locks = []
args.batch_maps_val = []
args.counter_maps_val = []
args.locks_val = []



# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def myexcepthook(exctype, value, traceback):
    for p in multiprocessing.active_children():
       p.terminate()


def main():
    start = time.time()
    args.gpu = 0 
    args.world_size = 1  
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
      args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.dist:
      args.distributed = True

    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.nnodes
    print("World:{}, nproc:{}, nnode:{}".format(dist_world_size, args.nproc_per_node, args.nnodes))


    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)
    args.world_size = dist_world_size

    processes = []
    lr_array = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
    if not args.distributed and args.num_jobs == 1:
        mp.spawn(main_worker, nprocs=args.nproc_per_node, args=(0, args, args.lr, args.master_port))
    #elif args.num_jobs == 1:
    #    main_worker(args.gpu, args.gpu, args, args.lr, args.master_port)
    else:
       #Hyperparameter search
       print("Running HP with {} jobs each on {} GPUS, num procs :{}".format(args.num_jobs, args.num_gpu_per_job, args.num_procs))
     
       # First create shared batch maps to store the minibatches
       # and their associated usage counters
       for job_procs in range(0, args.num_procs):
           args.batch_maps.append(multiprocessing.Manager().dict())
           args.counter_maps.append(multiprocessing.Manager().dict())
           args.locks.append(multiprocessing.Lock())
           args.batch_maps_val.append(multiprocessing.Manager().dict())
           args.counter_maps_val.append(multiprocessing.Manager().dict())
           args.locks_val.append(multiprocessing.Lock())
           
       if not args.dali_cpu:
           print("SETTING SYNC ON")       
       print("Batch maps:{}, counter_maps:{}".format(len(args.batch_maps), len(args.counter_maps)))

       for job in range(0, args.num_jobs):
           #The start ID of GPU for this multi-GPU HP job
           # Eg, if we start 4 2-GPU jobs, the IDs will be [0, 2, 4, 6]
           start_gpu_id = job*args.num_gpu_per_job
           #Change the port for each job - they are independent
           print("Before creating process - port is {} for job:{}".format(args.master_port, job))
           #master_port = args.master_port + job  
           #print("Spawn a {} GPU job starting at GPU#{}, port:{}".format(args.num_gpu_per_job, start_gpu_id, master_port))
           #processes.append(Process(target=main_worker, args=(0, start_gpu_id, args, lr_array[job], master_port)))
           processes.append(Process(target=spawn_jobs, args=(job, args, start_gpu_id, lr_array[job])))


    for process in processes:
        process.start()
        atexit.register(process.terminate)

    #while True:
    #    any_dead = False
    ##    for process in processes:
    #        if not process.is_alive():
    #            print("DEAD PROCESS {} with exitcode {}".format(process.pid, process.exitcode))
    #            any_dead = True
    #    if any_dead:
    #        break
    #    else:
    #        time.sleep(0.19)

    #for process in processes:
    #    if process.is_alive():
    #        print("silence {} I kill you at pid {}!".format(process.name, process.pid))
    #        process.terminate()


    #sys.excepth ok = myexcepthook
    for process in processes:
        print('%s.exitcode = %s' % (process.name, process.exitcode))
        process.join()

    dur = time.time() - start
    print("Total train time = {}".format(dur))


def spawn_jobs(job_id, args, start_id, lr):
    master_port = args.master_port + job_id
    print("Spawn a {} GPU job starting at GPU#{}, port:{}".format(args.num_gpu_per_job, start_id, master_port))
    mp.spawn(main_worker, nprocs=args.num_gpu_per_job, args=(start_id, args, lr, master_port))  

# gpu is the index of the job triggered by spawn method.
# If we spawn with nprocs=2, then two processes of index 
# 0 and 1 will be created.
def main_worker(gpu, start_gpu_id, args, lr, master_port):
    #This is the idx of the process in a multiGPU job
    args.local_rank_per_job=gpu
    args.master_port = master_port
    args.lr=lr
    start_full = time.time()
    #global args
    global best_prec1
 
    time_stat = []
    start = time.time()

    args.gpu = gpu + start_gpu_id
    args.dist_rank = args.nproc_per_node * args.node_rank + gpu
    args.local_rank = args.gpu
    print("{} MAIN WORKER job {} with LR:{} at PORT:{}".format(getpid(), args.gpu, args.lr, args.master_port))
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["RANK"] = str(args.dist_rank)
    os.environ["LOCAL_RANK"] = str(args.local_rank)


    #print("Process on GPU : {}".format(args.gpu)) 
    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
    else:
        torch.cuda.set_device(args.gpu) 
    args.total_batch_size = args.world_size * args.batch_size
    print("\nDistributed:{}\nGPU:{}\nWorld size:{}\nBatch:{}\nTotal batch:{}\nLocal rank:{}".format(args.distributed, args.gpu,args.world_size, args.batch_size, args.total_batch_size, args.local_rank))

    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if args.amp and args.fp16:
        print("Please use only one of the --fp16/--amp flags")
        exit(1)


    if args.static_loss_scale != 1.0:
        if not args.fp16:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")



    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    print("=> creating model AudioNet")
    model = audio_model.AudioNet(num_classes=args.classes)
    model = model.cuda()

    if args.fp16:
        model = network_to_half(model)


    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer,
            opt_level=args.opt_level,
            keep_batchnorm_fp32=args.keep_batchnorm_fp32,
            loss_scale=args.loss_scale,
            min_loss_scale=1.0
            )


    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # shared param/delay all reduce turns off bucketing in DDP, for lower latency runs this can improve perf
        # for the older version of APEX please use shared_param, for newer one it is delay_allreduce
        model = DDP(model, delay_allreduce=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    if len(args.data) == 1:
        traindir = os.path.join(args.data[0], 'train')
        valdir = os.path.join(args.data[0], 'val')
    else:
        traindir = args.data[0]
        valdir= args.data[1]

    if(args.arch == "inception_v3"):
        crop_size = 299
        val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        crop_size = 224
        val_size = 256

    print("Before calling Loader batch_maps={}".format(len(args.batch_maps)))
    if args.unified:
        pipe = AudioUnifiedTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=traindir, dali_cpu=args.dali_cpu, args_new=args)
        pipe.build()
        train_loader = DALIUnifiedAudioClassificationIterator(pipe, size=math.ceil(pipe.epoch_size("Reader") / args.num_procs), global_size=int(pipe.epoch_size("Reader")), last_batch_padded=True, auto_reset=True, batch_maps=args.batch_maps, counter_maps=args.counter_maps, locks=args.locks, iter_id=args.local_rank, must_pin=args.pin)
  
        if not args.noeval:
            pipe_val = AudioUnifiedValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=valdir,  dali_cpu=args.dali_cpu, args_new=args)
            pipe_val.build()
            val_loader = DALIUnifiedAudioClassificationIterator(pipe_val, size=math.ceil(pipe_val.epoch_size("Reader") / args.num_procs), global_size=int(pipe_val.epoch_size("Reader")), auto_reset=True, last_batch_padded=True, batch_maps=args.batch_maps_val, counter_maps=args.counter_maps_val, locks=args.locks_val, iter_id=args.local_rank, must_pin=args.pin)

    else:
        pipe = AudioTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=traindir,  dali_cpu=args.dali_cpu, args_new=args, shuffle_seed=args.local_rank)
        pipe.build()
        train_loader = DALIAudioClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

        if not args.noeval:
            pipe_val = AudioValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=valdir, dali_cpu=args.dali_cpu, args_new=args)
            pipe_val.build()
            val_loader = DALIAudioClassificationIterator(pipe_val, size=int(pipe_val.epoch_size("Reader") / args.world_size))

    if args.evaluate:
        validate(val_loader, model, args)
        return

    total_time = AverageMeter()
    dur_setup = time.time() - start
    time_stat.append(dur_setup)
    print("Batch size for GPU {} is {}, workers={}".format(args.gpu, args.batch_size, args.workers))

    for epoch in range(args.start_epoch, args.epochs):

        # log timing
        if args.local_rank == 0 and epoch == 0:
            os.system("swapoff -a")
            os.system("free -g") 
            print("STARTING EPOCH {}".format(epoch))
        start_ep = time.time()

        # train for one epoch

        avg_train_time = train(train_loader, model, optimizer, epoch, args)
        total_time.update(avg_train_time)
        #total_time.update(1)
        if args.prof:
            break
        # evaluate on validation set
        
        if args.noeval:
            [prec1, prec5] = [0,0]
        else:
            [prec1, prec5] = validate(val_loader, model, args)

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)
            if epoch == args.epochs - 1:
                print('##Top-1 {0}\n'
                      '##Top-5 {1}\n'
                      '##Perf  {2}'.format(prec1, prec5, args.total_batch_size / total_time.avg))


        scheduler.step()
        if not args.unified:
            # reset DALI iterators
            train_loader.reset()
            
            if not args.noeval:
                val_loader.reset()

        dur_ep = time.time() - start_ep
        print("Duration for epoch ={}".format(dur_ep))
        time_stat.append(dur_ep)

    #Cleanup loaders
    if args.unified:
        print("CALLING DELETE {}".format(args.local_rank))
        train_loader.shutdown()
        if not args.noeval:
            val_loader.shutdown()

    #if args.local_rank == 0:
    for i in time_stat:
        print("[{}]: Time_stat : {}".format(args.local_rank, i))

    for i in range(0, len(data_time_list)):
        print("[{}] Data time : {}\t Compute time : {}".format(args.local_rank, data_time_list[i], compute_time_list[i]))

    dur_full = time.time() - start_full
    #if args.local_rank == 0:
    print("[{}]Total time for all epochs = {}".format(args.local_rank, dur_full))    


def train(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    dataset_time = compute_time = 0

    for i, data in enumerate(train_loader):
        s = time.time() 
        audio = data[0]["data"].unsqueeze(1).cuda()    
        target = data[0]["label"].squeeze().cuda().long()   
        
        if args.unified:
            train_loader_len = int(math.ceil(train_loader._size*args.num_procs / args.batch_size))
        else:
            train_loader_len = int(math.ceil(train_loader._size / args.batch_size))
       
        if i == 10 and args.local_rank == 0:
            os.system("nvidia-smi")
            if epoch == 0:
                os.system("swapoff -a")
        if args.prof:
            if i > 10:
                break
        # measure data loading time
        #print("Batch:{}, Data:{}, get:{}, to_gpu:{}".format(i, time.time()-end, e1-end, time.time()-e1))
        #data_time.update(time.time() - s)
        data_time.update(time.time() - end)
        dataset_time += (time.time() - end)
        #dataset_time += (time.time() - s)
        compute_start = time.time()


        # compute output
        optimizer.zero_grad()
        audio = audio.requires_grad_() #set requires_grad to True for training
        output = model(audio)
        output = output.permute(1, 0, 2) #original output dimensions are batchSizex1xclasses
        loss = F.nll_loss(output[0], target) 

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            prec1 = reduce_tensor(prec1, args.world_size)
            prec5 = reduce_tensor(prec5, args.world_size)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), audio.size(0))
        top1.update(to_python_float(prec1), audio.size(0))
        top5.update(to_python_float(prec5), audio.size(0))

        if args.fp16:
            optimizer.backward(loss)
        elif args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        #if not args.dali_cpu:
        torch.cuda.synchronize()

        compute_time += (time.time() - compute_start)

        # measure elapsed time
        batch_time.update(time.time() - end)

        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0 and i > 1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {3:.3f} ({4:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, train_loader_len,
                   args.total_batch_size / batch_time.val,
                   args.total_batch_size / batch_time.avg,
                   batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
        #if i == 100:
            #sys.exit()
            #break
    if args.local_rank == 0:
        os.system("du -sh /dev/shm/cache")
 
    data_time_list.append(dataset_time)
    compute_time_list.append(compute_time)
    return batch_time.avg


def validate(val_loader, model, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, data in enumerate(val_loader):

        audio = data[0]["data"].unsqueeze(1).cuda()
        target = data[0]["label"].squeeze().cuda().long()
        if args.unified:
            val_loader_len = int(math.ceil(val_loader._size*args.num_procs / args.batch_size))
        else:
            val_loader_len = int(math.ceil(val_loader._size / args.batch_size))

        target = target.cuda(non_blocking=True)
 
        output = model(audio)    
        output = output.permute(1, 0, 2) 
        loss = F.nll_loss(output[0], target)  

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            prec1 = reduce_tensor(prec1, args.world_size)
            prec5 = reduce_tensor(prec5, args.world_size)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), audio.size(0))
        top1.update(to_python_float(prec1), audio.size(0))
        top5.update(to_python_float(prec5), audio.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, val_loader_len,
                   args.total_batch_size / batch_time.val,
                   args.total_batch_size / batch_time.avg,
                   batch_time=batch_time, 
                   top1=top1, top5=top5))
    
    print("Job:{}, LR:{}".format(args.local_rank, args.lr))
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return [top1.avg, top5.avg]


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, step, len_epoch, args):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr * (0.1 ** factor)

    """Warmup"""
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    if(args.local_rank == 0 and step % args.print_freq == 0 and step > 1):
        print("Epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = output.squeeze(0) 
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt

if __name__ == '__main__':
    main()
