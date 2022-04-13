# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import sys

import data
import model
import torch.utils.data.distributed
import torch.utils.data
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
#import torch.multiprocessing as mp
#mp.set_start_method('spawn')

sys.path.append('./')
from synergy_iterator import SynergyIterator

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100000,
                    help='upper epoch limit')
parser.add_argument('-b', '--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')

parser.add_argument("--sched-port", default='14000', type=str)
parser.add_argument("--sched-addr", default='127.0.0.1', type=str)
parser.add_argument('--job-id',  default=0, type=int)
parser.add_argument("--log-dir", default='./', type=str)
parser.add_argument('--noeval', action='store_true')
parser.add_argument('--synergy', action='store_true')
parser.add_argument("--max-iterations", default=-1, type=int)
parser.add_argument("--elapsed_iters", default=0, type=int)  
parser.add_argument("--chk_dir", default='./', type=str)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('-j', '--workers', default=3, type=int)

args = parser.parse_args()

torch.cuda.set_device(args.local_rank)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.


# Dataset Class so that we can use SynergyIterator
class LSTMDataset(torch.utils.data.Dataset):
    def __init__(self, data, bsz, bptt):
         # Work out how cleanly we can divide the dataset into bsz parts.
         nbatch = data.size(0) // bsz
         # Trim off any extra elements that wouldn't cleanly fit (remainders).
         self.data = data.narrow(0, 0, nbatch * bsz)
         # Evenly divide the data across the bsz batches.
         self.data = self.data.view(bsz, -1).t().contiguous().to(device)
         self.batch_size = bsz
         self.data_len = self.data.size(0)  
         self.bptt = bptt


    # get_batch subdivides the source data into chunks of length args.bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.

    def get_batch(self, i, j):
        row_id = i % len(self.data)
        seq_len = min(self.bptt, len(self.data) - 1 - row_id)
        data = self.data[row_id:row_id+seq_len, j]
        target = self.data[row_id+1:row_id+1+seq_len, j].view(data.size())
        data = torch.cat([data, data.new_zeros(self.bptt - data.size(0))])
        target = torch.cat([target, target.new_zeros(self.bptt - target.size(0))])
        #print("i={}, Row id ={}, j={}, data len={}, self.data_len={}".format(i,row_id, j, len(self.data), self.data_len))
        #print("Seq len = {}, data dim={}. tgt dim={}".format(seq_len, data.dim(), target.dim()))
        return data, target

    #def get_batch(self, i):
    #    seq_len = min(self.bptt, len(self.data) - 1 - i)
    #    data = self.data[i:i+seq_len]
    #    target = self.data[i+1:i+1+seq_len].view(-1)
    #    return data, target


    def __len__(self):
        #return self.data_len // self.bptt
        return self.data_len // self.bptt * self.batch_size
        #return self.data_len

    def __getitem__(self, idx):
        #print("Returning for idx {}, i={}".format(idx, idx*self.bptt))
        #return self.get_batch(idx*self.bptt, idx%self.batch_size)
        return self.get_batch((idx // self.batch_size)*self.bptt, idx % self.batch_size)


#def batchify(data, bsz):
#    # Work out how cleanly we can divide the dataset into bsz parts.
#    nbatch = data.size(0) // bsz
#    # Trim off any extra elements that wouldn't cleanly fit (remainders).
#    data = data.narrow(0, 0, nbatch * bsz)
#    # Evenly divide the data across the bsz batches.
#    data = data.view(bsz, -1).t().contiguous()
#    return data.to(device)

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)



def evaluate(val_loader):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            data, targets = batch 
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(val_loader) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        if args.distributed:
            hidden = model.module.init_hidden(args.batch_size)
        else:
            hidden = model.init_hidden(args.batch_size)
    print("Starting train")
    for i, batch in enumerate(train_loader):
        #print("Processing batch {}/{}".format(i, len(train_loader)))
        (data, targets) = batch
        data = data.t()
        targets = targets.t()
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        args.steps_so_far += 1
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets.flatten())
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad)

        total_loss += loss.item()

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, i, len(train_loader), lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


def save_checkpoint(model):
    print("Saving checkpoint to {}".format(args.save))
    chk = {
        'model': model,
        'steps_so_far': args.steps_so_far,
        'epoch': args.epoch,
       }
    with open(args.save, 'wb') as f:
        torch.save(chk, f) 


os.environ["SYNERGY_JOB_ID"] = str(args.job_id)
os.environ["SYNERGY_SCHED_ADDR"] = args.sched_addr
os.environ["SYNERGY_SCHED_PORT"] = args.sched_port
os.environ["SYNERGY_LOG_DIR"] = args.log_dir
os.environ["SYNERGY_DEBUG"] = "true"

# Loop over epochs.
lr = args.lr
best_val_loss = None
args.steps_so_far = 0
args.epoch = 0
args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if args.distributed:
    args.gpu = args.local_rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(backend='nccl',
                                        init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    

eval_batch_size = 10
#train_data = batchify(corpus.train, args.batch_size)
#val_data = batchify(corpus.valid, eval_batch_size)
#test_data = batchify(corpus.test, eval_batch_size)

train_data = LSTMDataset(corpus.train, args.batch_size, args.bptt)
val_data = LSTMDataset(corpus.valid, eval_batch_size, args.bptt)
test_data = LSTMDataset(corpus.test, eval_batch_size, args.bptt)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
if args.model == 'Transformer':
    model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.NLLLoss()

if args.distributed:
    model = DDP(model)


if not os.path.isdir(args.chk_dir):
    os.makedirs(args.chk_dir)

#Load chk here
args.save = os.path.join(args.chk_dir, 'model.chk')
print("CHK Path = {}".format(args.save))
args.steps_so_far = args.elapsed_iters
if False and os.path.exists(args.save):
    if os.path.isfile(args.save):
        with open(args.save, 'rb') as f:
           checkpoint = torch.load(f)
           model = checkpoint['model']
           args.steps_so_far = checkpoint['steps_so_far']
           args.epoch = checkpoint['epoch']

if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
else:
    train_sampler = None
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, 
                                sampler=train_sampler, drop_last=True)

if not args.noeval:
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=eval_batch_size, shuffle=False, 
                                drop_last=True )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=eval_batch_size, shuffle=False, 
                                drop_last=True )

if args.synergy:
    train_loader = SynergyIterator(train_loader,max_iters=args.max_iterations, iters_elapsed=args.steps_so_far)
print("TL len after={}".format(len(train_loader))) 

if args.max_iterations > 0:
    args.epochs = math.ceil(args.max_iterations / len(train_loader))

print("Num epochs = {}".format(args.epochs))

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(args.epoch, args.epochs+1):
        epoch_start_time = time.time()
        train()
        if args.synergy:
            print("Finished train loop once, steps={}, exit={}".format(train_loader._total_steps, train_loader._exit))
        else:
            print("Finished train loop once")
        if args.synergy and train_loader.exit:
            print("Break now")
            break

        if not args.noeval:
            val_loss = evaluate(val_loader)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0

    if args.local_rank == 0:
        save_checkpoint(model)

    if args.synergy:
        train_loader.complete()

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

if not args.noeval:
    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(test_loader)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    if len(args.onnx_export) > 0:
        # Export the model in ONNX format.
        export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
