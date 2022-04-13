#!/bin/bash

#if [ "$#" -ne 3 ]; then
#	echo "Usage : ./run-1-job.sh <data-dir> <out-dir> <worker>"
#	exit 1
#fi

#DATA_DIR=$1
#OUT_DIR=$2
WORKER=3



echo " Data dir is $DATA_DIR"
echo " Out dir is $OUT_DIR"


arch='transformer'
batch=128

      job_name="${arch}_${batch}_2"
      start=$SECONDS
      python profiler/offline_profiler.py --job-name $job_name --cpu 24 --num-gpus 2 -b $batch --docker-img=fiddlev3.azurecr.io/synergy_dali:latest ../models/Transformer-XL/pytorch/train.py  --max-iterations 100 --config_file '../models/Transformer-XL/pytorch//wt103_base.yaml'  --synergy --noeval --src '../models/Transformer-XL/pytorch/' --data  '/datadrive/mnt2/jaya/datasets/wikitext-103'
      duration=$(( SECONDS - start ))
      echo "RAN $arch for $batch for $duration s"
