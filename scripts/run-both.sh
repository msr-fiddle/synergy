#!/bin/bash

OUT_DIR=$1
WORKER=$2
outfile="all-utils-${WORKER}.csv"
dstat -cdnmgyr --output $outfile 2>&1 &  
./audio_classification/run-m5-4gpu.sh /datadrive/mnt6/jaya/datasets/fma/ $OUT_DIR/m5/ $WORKER
pkill -f dstat
mv $outfile $OUT_DIR/m5

echo 3 > /proc/sys/vm/drop_caches
outfile="all-utils-${WORKER}.csv"
dstat -cdnmgyr --output $outfile 2>&1 &  
./image_classification/run-res18-4gpu.sh  /datadrive/mnt6/jaya/datasets/openimages/ $OUT_DIR/res/ $WORKER
pkill -f dstat 
mv $outfile $OUT_DIR/res/
