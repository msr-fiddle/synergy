#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage : ./run.sh <out-dir>"
    exit 1
fi

OUT_DIR=$1
#WORKERS=3
#for MEM in 150 200 80 120 140; do
for WORKERS in 1 2 24 12 3 6; do
#OUT_DIR=$1/${MEM}/
#for WORKERS in 24 16 12 6 3 2 1; do
#nvidia-docker run --ipc=host --mount src=/,target=/datadrive/,type=bind --rm -it --network=host --cpus="24" --cpuset-cpus=0-23 --privileged j-bert /bin/bash -c 'pwd; ls /datadrive; cd /datadrive/mnt2/jaya/scheduler/BERT/; '
    nvidia-docker run --ipc=host --mount src=/,target=/datadrive/,type=bind --rm -it --network=host --cpus="$WORKERS" --cpuset-cpus=0-$(($WORKERS-1)) --privileged -e "OUT_DIR="$OUT_DIR"" -e "WORKER="$WORKERS"" j-deepspeech /bin/bash -c 'cd /datadrive/mnt2/jaya/scheduler/deepspeech.pytorch; ./run-1gpu.sh "$OUT_DIR" "$WORKER"'
#    nvidia-docker run --ipc=host --mount src=/,target=/datadrive/,type=bind --rm -it --network=host --cpus="$WORKERS" --cpuset-cpus=0-$(($WORKERS-1)) -m "$MEM"g --privileged -e "OUT_DIR="$OUT_DIR"" -e "WORKER="$WORKERS"" fiddlev3.azurecr.io/dali_py36_cu10_pytorch  /bin/bash -c 'cd /datadrive/mnt2/jaya/scheduler/; echo 3 > /proc/sys/vm/drop_caches; ./scripts/run-1job.sh /datadrive/mnt1/deepak/data/imagenet/ "$OUT_DIR" "$WORKER"'
done
