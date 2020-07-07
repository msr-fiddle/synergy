#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python -m torch.distributed.launch --nproc_per_node="$2" train.py \
        --cuda \
        --data ../data/one-billion-words/ \
        --dataset lm1b \
        --adaptive \
        --n_layer 18 \
        --d_model 1024 \
        --div_val 4 \
        --n_head 8 \
        --d_head 128 \
        --d_inner 4096 \
        --dropout 0.0 \
        --dropatt 0.0 \
        --optim adam \
        --warmup_step 20000 \
        --max_step 500000 \
        --lr 0.00025 \
        --tgt_len 32 \
        --mem_len 32 \
        --eval_tgt_len 32 \
        --batch_size 128 \
        #--batch_size 224 \
        --multi_gpu ddp\
        #--gpu0_bsz 32 \
        ${@:3}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python -m torch.distributed.launch --nproc_per_node="$2" eval.py \
        --cuda \
        --data ../data/one-billion-words/ \
        --dataset lm1b \
        --batch_size 64 \
        --tgt_len 32 \
        --mem_len 128 \
        --split test \
        --same_length \
        ${@:3}
else
    echo 'unknown argment 1'
fi
