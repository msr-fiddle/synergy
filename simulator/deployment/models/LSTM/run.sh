#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage : ./run.sh <out-dir>"
    exit 1
fi

OUT_DIR=$1

#for WORKERS in 24; do
for WORKERS in 1 3 12 6 2; do
#for WORKERS in 24 16 12 6 3 2 1; do
    nvidia-docker run --ipc=host --mount src=/,target=/datadrive/,type=bind --rm -it --network=host --cpus="$WORKERS" --cpuset-cpus=0-$(($WORKERS-1)) --privileged -e "OUT_DIR="$OUT_DIR"" -e "WORKER="$WORKERS"" j-bert /bin/bash -c 'cd /datadrive/mnt2/jaya/scheduler/LSTM/; ./run_lstm.sh  "$OUT_DIR" "$WORKER"'
done
