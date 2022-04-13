import argparse
import os, errno
import shutil
import time
import math
import sys 

from synergy_iterator import SynergyIterator
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


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training using DALI')
parser.add_argument('--data', metavar='DIR', default="./", type=str, #nargs='*',
                    help='path(s) to dataset (if one path is provided, it is assumed\n' +
                    'to have subdirectories named "train" and "val"; alternatively,\n' +
                    'train and val paths can be specified directly by providing both paths as arguments)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')

parser.add_argument('--job-name',  default="job-0", type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100000, type=int, metavar='N',
                    help='number of total epochs to run')
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
parser.add_argument('--nopin', action='store_false', help='Use this '  
                                 'argument to disable memory pinning')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--chk_dir', default='./chk/', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: ./chk)')
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

parser.add_argument("--max-iterations", default=-1, type=int)
parser.add_argument("--elapsed_iters", default=0, type=int)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--classes", default=1000, type=int)
parser.add_argument("--cache_size", default=0, type=int)
parser.add_argument('--sync_bn', action='store_true',
        help='enabling apex sync BN.')
parser.add_argument('--opt-level', type=str)
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)
parser.add_argument('--channels-last', type=bool, default=False)
parser.add_argument('--deterministic', action='store_true')
parser.add_argument('--noeval', action='store_true')
parser.add_argument('--amp',action='store_true',help='Run model AMP (automatic mixed precision) mode.')
parser.add_argument("--nnodes", default=1, type=int)
parser.add_argument("--node_rank", default=0, type=int)
parser.add_argument("--shuffle_seed", default=0, type=int)
parser.add_argument('--mint', action='store_true')
parser.add_argument('--dali', action='store_true')
parser.add_argument('--node_ip_list', action='append', type=str, help='Enter IP of other nodes in order')  
parser.add_argument('--node_port_list', action='append', type=int, help='Enter start port of other nodes in order') 
parser.add_argument('--synergy', action='store_true')
parser.add_argument('--step-lr', action='store_true')
parser.add_argument('--job-id',  default=0, type=int)
parser.add_argument("--sched-addr", default='127.0.0.1', type=str)
parser.add_argument("--sched-port", default='14000', type=str)
parser.add_argument("--log-dir", default='./', type=str)

cudnn.benchmark = True

compute_time_list = []
data_time_list = []

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, resume_index=0, resume_epoch=0):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)

        shard = args.global_rank
        #shard = int(args.node_rank*args.world_size/args.nnodes + args.local_rank)
        if args.mint:
            self.input = ops.FileReader(file_root=data_dir, shard_id=shard, num_shards=args.world_size, shuffle_after_epoch=True, cache_size=args.cache_size, shuffle_seed=args.shuffle_seed, debug=True)
        else:
            #if not resume_index and not resume_epoch:
            synergy_det=True
            print("SYNERGY DETERMINISTIC SHUFFLING is {}".format(synergy_det))
            self.input = ops.FileReader(file_root=data_dir, shard_id=shard, num_shards=args.world_size, shuffle_after_epoch=True, shuffle_seed=args.shuffle_seed, debug=True, resume_index=resume_index, resume_epoch=resume_epoch, synergy_det=synergy_det)
        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        #decoder_device = 'cpu' 
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        #shard = int(args.node_rank*args.world_size/args.nnodes + args.local_rank)
        shard = args.global_rank
        self.input = ops.FileReader(file_root=data_dir, shard_id=shard, num_shards=args.world_size, random_shuffle=False)
        self.decode = ops.ImageDecoder(device="cpu", output_type=types.RGB)
        self.res = ops.Resize(device="cpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu())
        return [output, self.labels]

best_prec1 = 0
args = parser.parse_args()

#if args.local_rank == 0:
#    register_handle = Register(name=args.job_name)
print(args.data)

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

if args.amp:
    print("Using mixed precision : {}".format(args.amp))
    print("opt_level = {}".format(args.opt_level))
    print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))

if args.dali:
    print("Using DALI")
else:
    print("Using native dataloader")
args.global_rank = 0
args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if args.distributed:
    args.global_rank = os.environ['RANK']
    print("WORLD={}, rank={}, global_rank={}".format(os.environ['WORLD_SIZE'], os.environ['LOCAL_RANK'], os.environ['RANK']))

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def main():
    start_full = time.time()
    global best_prec1, args

    time_stat = []
    acc_list = []
    time_list = []
    start = time.time()

    
    args.gpu = 0
    args.world_size = 1
    torch.cuda.set_device(args.gpu)

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    args.total_batch_size = args.world_size * args.batch_size

    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if args.amp and args.fp16:
        print("Please use only one of the --fp16/--amp flags")
        exit(1)


    if args.static_loss_scale != 1.0:
        if not args.fp16:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    try:
        os.makedirs(args.chk_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if(args.arch == "inception_v3"):
            model = models.__dict__[args.arch](num_classes=args.classes,aux_logits=False)
        else:
            model = models.__dict__[args.arch](num_classes=args.classes)

    model = model.cuda()

    if args.fp16:
        model = network_to_half(model)

    args.lr = args.lr*float(args.batch_size*args.world_size)/256.
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
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
    args.start_index = 0
    args.steps_so_far = 0
    chk_path = os.path.join(args.chk_dir, 'model.chk')
    print("CHK Path = {}".format(chk_path))
    args.steps_so_far = args.elapsed_iters
    # For now, dont oad from checkpoints
    if False and os.path.exists(chk_path):
        args.resume = True
        if os.path.isfile(chk_path):
            print("=> loading checkpoint '{}'".format(chk_path))
            checkpoint = torch.load(chk_path, map_location=lambda storage, loc: storage.cuda(args.gpu))
            args.start_epoch = checkpoint['epoch']
            args.start_index = checkpoint['iter']*args.batch_size
            args.steps_so_far = checkpoint['steps_so_far']
            args.shuffle_seed = checkpoint['dl_shuffle_seed']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}, iter {}, step {})"
                  .format(chk_path, checkpoint['epoch'], checkpoint['iter'], checkpoint['steps_so_far']))
        else:
            print("=> no checkpoint found at '{}'".format(chk_path))

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    train_pipe = None

    if args.dali:
        if(args.arch == "inception_v3"):
            crop_size = 299
            val_size = 320 # I chose this value arbitrarily, we can adjust.
        else:
            crop_size = 224
            val_size = 256

        if not args.synergy:
            args.start_index = 0
            pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=traindir, crop=crop_size, dali_cpu=args.dali_cpu)
        else:
            pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=traindir, crop=crop_size, dali_cpu=args.dali_cpu, resume_index=args.start_index, resume_epoch=args.start_epoch)

        pipe.build()
        train_pipe = pipe

        #number of samples for this resumed epoch. If full epoch, resume_size = size
        resume_size = int(pipe.epoch_size("Reader") / args.world_size) - args.start_index 
        print("Train Loader size = {}, resume_size = {}".format(pipe.epoch_size("Reader") / args.world_size, resume_size)) 
        train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size), fill_last_batch=False, resume_size=resume_size)
        if  args.synergy:
            train_loader = SynergyIterator(train_loader, dali=args.dali, max_iters=args.max_iterations, iters_elapsed=args.steps_so_far)

        if not args.noeval:
            pipe_val = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=valdir, crop=crop_size, size=val_size)
            pipe_val.build()
            val_loader = DALIClassificationIterator(pipe_val, size=int(pipe_val.epoch_size("Reader") / args.world_size), fill_last_batch=False)

    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
           traindir,
           transforms.Compose([
             transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalize,
        ]))
        if args.distributed:
           train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
           train_sampler = None
        train_loader = torch.utils.data.DataLoader(
             train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=args.nopin, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
               transforms.Resize(256),
               transforms.CenterCrop(224),
               transforms.ToTensor(),
               normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=args.nopin)


    if args.evaluate and not args.noeval:
        validate(val_loader, model, criterion)
        return

    total_time = AverageMeter()
    dur_setup = time.time() - start
    time_stat.append(dur_setup)
    print("Batch size for GPU {} is {}, workers={}".format(args.gpu, args.batch_size, args.workers))

    for epoch in range(args.start_epoch, args.epochs):

        if args.local_rank == 0 and epoch == 0:
            os.system("swapoff -a")
            os.system("free -g") 
             
        # log timing
        start_ep = time.time()

        # train for one epoch
        avg_batch_time, iteration = train(train_loader, model, criterion, optimizer, epoch)
        print("Train returned : {}".format(iteration))
        

        total_time.update(avg_batch_time)

        if args.prof:
            break



        # evaluate on validation set if epoch boundary
        [prec1, prec5] = [0,0]
        if iteration == 0:
            if args.noeval:
                [prec1, prec5] = [0,0]
            else:
                [prec1, prec5] = validate(val_loader, model, criterion)
            acc_list.append(prec1)


        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            #Epoch starts from 0
            if iteration > 0:
                epoch_save = epoch
            else:
                epoch_save = epoch + 1
            save_checkpoint({
                'epoch': epoch_save,
                'arch': args.arch,
                'iter': iteration,
                'steps_so_far': args.steps_so_far,
                'dl_shuffle_seed': args.shuffle_seed,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename=chk_path)
            
            if epoch == args.epochs - 1:
                print('##Top-1 {0}\n'
                      '##Top-5 {1}\n'
                      '##Perf  {2}'.format(prec1, prec5, args.total_batch_size / total_time.avg))


        # EIther end of job, or lease
        if args.synergy and train_loader.exit:
            break

        dur_ep = time.time() - start_ep
        print("EPOCH DURATION = {}".format(dur_ep))
        time_stat.append(dur_ep)

        if iteration == 0:
            time_list.append(dur_ep)


        if args.dali:
            # reset DALI iterators
            train_loader.reset()
            if not args.noeval:
                val_loader.reset()

    # Exiting the train loop, so cleanup
    if args.synergy:
        train_loader.complete()

    if args.local_rank == 0:
        for i in time_stat:
            print("Time_stat : {}".format(i))

        for i in range(0, len(data_time_list)):
            print("Data time : {}\t Compute time : {}".format(data_time_list[i], compute_time_list[i]))

    dur_full = time.time() - start_full
    if args.local_rank == 0:
        print("Total time for all epochs = {}".format(dur_full))   
    

    if args.dali:
        del pipe
        if not args.noeval:
            del pipe_val 

    return total_time.avg


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    dataset_time = compute_time = 0
    i = -1
    for i, data in enumerate(train_loader):
        if args.dali:
            images = data[0]["data"]
            target = data[0]["label"].squeeze().cuda().long()
            train_loader_len = int(math.ceil(train_loader._size / args.batch_size))
            input_var = Variable(images)
            target_var = Variable(target)

        else:
            images, target = data
            target = target.squeeze().cuda().long()
            input_var = Variable(images).cuda(args.gpu, non_blocking=True)
            target_var = Variable(target).cuda(args.gpu, non_blocking=True)
            train_loader_len =  int(len(train_loader))

        adjust_learning_rate(optimizer, epoch, i, train_loader_len, args.steps_so_far)
       
        #if i == 10:
        #    os.system("nvidia-smi")
   
        if args.prof:
            if i > 10:
                break

        # measure data loading time
        data_time.update(time.time() - end)
        dataset_time += (time.time() - end)
        compute_start = time.time()


        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), images.size(0))
        top1.update(to_python_float(prec1), images.size(0))
        top5.update(to_python_float(prec5), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss)
        elif args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        torch.cuda.synchronize()
        compute_time += (time.time() - compute_start)
        
        #args.steps_so_far += 1
        # measure elapsed time
        if i > 10:		
            batch_time.update(time.time() - end)

        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0 and i > 10:
            #register_handle.log_update(batch_time.avg, i, epoch)
      
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
    data_time_list.append(dataset_time)
    compute_time_list.append(compute_time)
    if i == -1:
        return batch_time.avg, 0

    args.steps_so_far += (i + 1)
    return batch_time.avg, (i+1) % train_loader_len


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, data in enumerate(val_loader):
        if args.dali:
            images = data[0]["data"]
            target = data[0]["label"].squeeze().cuda().long()
            val_loader_len = int(val_loader._size / args.batch_size)

            target = target.cuda(non_blocking=True)

            input_var = Variable(images)
            target_var = Variable(target)
        else:
            images, target = data 
            target = target.squeeze().cuda().long()   
            val_loader_len = int(len(val_loader))
            input_var = Variable(images).cuda(args.gpu, non_blocking=True)
            target_var = Variable(target).cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), images.size(0))
        top1.update(to_python_float(prec1), images.size(0))
        top5.update(to_python_float(prec5), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, val_loader_len,
                   args.total_batch_size / batch_time.val,
                   args.total_batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return [top1.avg, top5.avg]


def save_checkpoint(state, is_best, filename='model.chk'):
    print("Checkpointing at {}".format(filename))
    torch.save(state, filename)
    #if is_best:
    #    shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch, step, len_epoch, steps_so_far):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    if args.step_lr:
        # apply lr schedule based on steps completed
        epoch = int(steps_so_far/len_epoch)
        step = step + (steps_so_far % len_epoch)

    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr * (0.1 ** factor)

    """Warmup"""
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    if(args.local_rank == 0 and step % args.print_freq == 0 and step > 1):
        if not args.step_lr:
            print("Epoch = {}, step = {}, lr = {}".format(epoch, step, lr))
        else:
            print("STEP LR : Epoch = {}, step = {}, lr = {}, steps_so_far={}".format(epoch, step, lr, steps_so_far))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':

    os.environ["SYNERGY_JOB_ID"] = str(args.job_id)
    os.environ["SYNERGY_SCHED_ADDR"] = args.sched_addr
    os.environ["SYNERGY_SCHED_PORT"] = args.sched_port
    os.environ["SYNERGY_LOG_DIR"] = args.log_dir
    os.environ["SYNERGY_DEBUG"] = "true"
    main()
