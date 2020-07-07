#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage : ./run.sh <out-dir>"
    exit 1
fi

OUT_DIR=$1

for WORKERS in 16 12 6 3 2 1; do
#for WORKERS in 24 16 12 6 3 2 1; do
#nvidia-docker run --ipc=host --mount src=/,target=/datadrive/,type=bind --rm -it --network=host --cpus="24" --cpuset-cpus=0-23 --privileged j-bert /bin/bash -c 'pwd; ls /datadrive; cd /datadrive/mnt2/jaya/scheduler/BERT/; '
#    nvidia-docker run --ipc=host --mount src=/,target=/datadrive/,type=bind --rm -it --network=host --cpus="$WORKERS" --cpuset-cpus=0-$(($WORKERS-1)) --privileged -e "OUT_DIR="$OUT_DIR"" j-bert /bin/bash -c 'echo "$OUT_DIR"'
    nvidia-docker run --ipc=host --mount src=/,target=/datadrive/,type=bind --rm -it --network=host --cpus="$WORKERS" --cpuset-cpus=0-$(($WORKERS-1)) --privileged -e "OUT_DIR="$OUT_DIR"" -e "WORKER="$WORKERS"" j-bert /bin/bash -c 'cd /datadrive/mnt2/jaya/scheduler/BERT/; ./run_pretraining_phase1.sh /datadrive/mnt4/jaya/datasets/BERT/ "$OUT_DIR" "$WORKER"'
done
