#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage : ./run-1-job.sh <out-dir> <worker>"
  exit 1
fi

OUT_DIR=$1
WORKERS=$2
SCRIPTS="../scripts/"
batch=64
num_gpu=1

mkdir -p $OUT_DIR  
result_dir="${OUT_DIR}/deepspeech_b${batch}_w${WORKERS}_g${num_gpu}"
mkdir -p $result_dir 

./$SCRIPTS/free.sh &  
./$SCRIPTS/gpulog.sh &   
mpstat -P ALL 1 > cpu_util.out 2>&1 &  
dstat -cdnmgyr --output all-utils.csv 2>&1 &
python train.py --rnn-type lstm --hidden-size 1024 --hidden-layers 5  --train-manifest data/libri_train_manifest.csv --val-manifest data/libri_val_manifest.csv --epochs 2 --num-workers $WORKERS --cuda  --learning-anneal 1.01 --batch-size $batch --no-sortaGrad  --opt-level O1 --loss-scale 1 --id libri 2>&1 > stdout.out  

pkill -f dstat
pkill -f mpstat 
pkill -f free  
pkill -f gpulog 
pkill -f nvidia-smi 

sleep2
mv *.out  $result_dir/  
mv *.log $result_dir/  
mv *.csv $result_dir/ 
