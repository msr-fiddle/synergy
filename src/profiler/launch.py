r"""
`Adapted from torch.distributed.launch`. 

This module spawns up multiple distributed
training processes on each of the training nodes
or GPUs (for data-parallilism).

This utility assumes that the training script input by the user
trains on a single GPU (GPU id is given by 'local_rank').
The batch_size, CPU workers etc are assumed to be set per GPU
in the training script. This module will not modify 
such parameters. FOr a multi-GPU training in a single node,
this module will spawn n processes, one on each GPU.

This utility shares the same input args as torch.distr.launch
`--nproc_per_node=n` is the number of GPU processes per node
	uses GPUs 0 to n by default unless `gpu_list` is set
	Scheduler must ensure that this is < # GPUs available
	node. 
`--gpu_list=0,1,2` is the list of GPU IDs to use for
	training. If not set, the first n GPUs starting
	with 0 are used.

`--master_addr` is the IP of the master for multi-node
	training. Defaults to localhost 127.0.0.1

`--master_port` is the free port on master. Defaults
	to 

`--nnodes` is the number of nodes involved in training
	Defaults to 1

`--node_rank` is the global rank of the node among all
	used in training. Goes from 0 to nnodes-1
	Defaults to 0

**How to use this module:**

1. Single-Node multi-process distributed training


    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
               arguments of your training script)

2. Multi-Node multi-process distributed training: (e.g. two nodes)


Node 1: *(IP: 192.168.1.1, and has a free port: 1234)*

::

    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
               --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)

Node 2:

::

    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
               --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)

3. To look up what optional arguments this module offers:

::

    >>> python -m torch.distributed.launch --help


**Important Notices:**

1. This utilty and multi-process distributed (single-node or
multi-node) GPU training currently only achieves the best performance using
the NCCL distributed backend. Thus NCCL backend is the recommended backend to
use for GPU training.

2. In your training program, you must parse the command-line argument:
``--local_rank=LOCAL_PROCESS_RANK``, which will be provided by this module.
If your training program uses GPUs, you should ensure that your code only
runs on the GPU device of LOCAL_PROCESS_RANK. This can be done by:

Parsing the local_rank argument

::

    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument("--local_rank", type=int)
    >>> args = parser.parse_args()

Set your device to local rank using either

::

    >>> torch.cuda.set_device(arg.local_rank)  # before your code runs

or

::

    >>> with torch.cuda.device(arg.local_rank):
    >>>    # your code to run

3. In your training program, you are supposed to call the following function
at the beginning to start the distributed backend. You need to make sure that
the init_method uses ``env://``, which is the only supported ``init_method``
by this module.

::

    torch.distributed.init_process_group(backend='YOUR BACKEND',
                                         init_method='env://')

4. In your training program, you can either use regular distributed functions
or use :func:`torch.nn.parallel.DistributedDataParallel` module. If your
training program uses GPUs for training and you would like to use
:func:`torch.nn.parallel.DistributedDataParallel` module,
here is how to configure it.

::

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[arg.local_rank],
                                                      output_device=arg.local_rank)

Please ensure that ``device_ids`` argument is set to be the only GPU device id
that your code will be operating on. This is generally the local rank of the
process. In other words, the ``device_ids`` needs to be ``[args.local_rank]``,
and ``output_device`` needs to be ``args.local_rank`` in order to use this
utility

5. Another way to pass ``local_rank`` to the subprocesses via environment variable
``LOCAL_RANK``. This behavior is enabled when you launch the script with
``--use_env=True``. You must adjust the subprocess example above to replace
``args.local_rank`` with ``os.environ['LOCAL_RANK']``; the launcher
will not pass ``--local_rank`` when you specify this flag.

.. warning::

    ``local_rank`` is NOT globally unique: it is only unique per process
    on a machine.  Thus, don't use it to decide if you should, e.g.,
    write to a networked filesystem.  See
    https://github.com/pytorch/pytorch/issues/12042 for an example of
    how things can go wrong if you don't do this correctly.

"""


import sys
import subprocess
import os
from argparse import ArgumentParser, REMAINDER


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="PyTorch distributed training launch "
                                        "helper utilty that will spawn up "
                                        "multiple distributed processes")

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

    # positional
    parser.add_argument("training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.nnodes

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)

    processes = []

    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        if args.use_env:
            cmd = [sys.executable, "-u",
                   args.training_script] + args.training_script_args
        else:
            cmd = [sys.executable,
                   "-u",
                   args.training_script,
                   "--local_rank={}".format(local_rank), 
				   "--nnodes={}".format(args.nnodes),
				   "--node_rank={}".format(args.node_rank)] + args.training_script_args

        #process = subprocess.check_output(cmd, env=current_env)
       	#print("Return = {}".format(process))
        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        print("Return = {}".format(process.returncode))
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=process.args)


if __name__ == "__main__":
    main()
