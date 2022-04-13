#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage : ./run.sh <out-dir> <sched_ip> <sched_port> <job_id>"
    exit 1
fi

OUT_DIR=$1/50g/
SCHED_IP=$2
SCHED_PORT=$3
JOB_ID=$4
WORKERS=12
work=$((WORKERS / 4))
nvidia-docker run --ipc=host --mount src=/,target=/datadrive/,type=bind --rm -it --network=host --cpus="$WORKERS" --cpuset-cpus=0-$(($WORKERS-1)) -m 50g --privileged -e "SCHED_PORT="$SCHED_PORT"" -e "SCHED_IP="$SCHED_IP"" -e "JOB_ID="$JOB_ID"" nvidia/dali:py36_cu10.run /bin/bash -c 'cd /datadrive/mnt2/jaya/scheduler-sim/python-simulator/deployment; python dummy_job_script.py "$SCHED_IP" "$SCHED_PORT" "$JOB_ID"'
