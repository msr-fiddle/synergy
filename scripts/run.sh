#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage : ./run.sh <out-dir>"
    exit 1
fi

OUT_DIR=$1/450g/

for WORKERS in 12 24; do
    work=$((WORKERS / 4))
    nvidia-docker run --ipc=host --mount src=/,target=/datadrive/,type=bind --rm -it --network=host --cpus="$WORKERS" --cpuset-cpus=0-$(($WORKERS-1)) -m 450g --privileged -e "OUT_DIR="$OUT_DIR"" -e "WORKER="$work"" nvidia/dali:py36_cu10.run /bin/bash -c 'cd /datadrive/mnt2/jaya/scheduler/; ./run-both.sh  "$OUT_DIR" "$WORKER"'
done

exit
OUT_DIR=$1/250g/

for WORKERS in 24; do
    work=$((WORKERS / 4))
    nvidia-docker run --ipc=host --mount src=/,target=/datadrive/,type=bind --rm -it --network=host --cpus="$WORKERS" --cpuset-cpus=0-$(($WORKERS-1)) -m 450g --privileged -e "OUT_DIR="$OUT_DIR"" -e "WORKER="$work"" nvidia/dali:py36_cu10.run /bin/bash -c 'cd /datadrive/mnt2/jaya/scheduler/; ./run-both.sh  "$OUT_DIR" "$WORKER"'
done
