#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage : ./run.sh <out-dir> <sched_ip> <sched_port> <job_id>"
    exit 1
fi

OUT_DIR=$1/50g/
SCHED_IP=$2
SCHED_PORT=$3
JOB_ID=$4

WORKERS=${SYNERGY_CPU_THIS_SERVER%.*}
MEM=${SYNERGY_MEM_THIS_SERVER%.*}
ITERS=${SYNERGY_TOTAL_ITERS%.*}
ARCH=$SYNERGY_ARCH
BATCH=${SYNERGY_BATCH_SIZE%.*}

echo $WORKERS, $MEM, $ITERS, $ARCH, $BATCH

nvidia-docker run \
	--ipc=host \
	--mount src=/,target=/datadrive/,type=bind --rm -it \
	--network=host \
	--cpus="$WORKERS" --cpuset-cpus=0-$(($WORKERS-1)) \
	-m "$MEM"g \
	--privileged \
	-e "SCHED_PORT="$SCHED_PORT"" \
	-e "SCHED_IP="$SCHED_IP"" \
	-e "JOB_ID="$JOB_ID""  \
	-e "ITERS="$ITERS"" \
	-e "ARCH="$ARCH"" \
	-e "BATCH="$BATCH"" \
	-e "WORKERS="$WORKERS"" \
	nvidia/dali:py36_cu10.run /bin/bash \
	-c 'cd /datadrive/mnt2/jaya/scheduler-sim/python-simulator/deployment; \
	python job_script.py \
		--sched-addr "$SCHED_IP" \
		--sched-port "$SCHED_PORT" \
		--job-id "$JOB_ID" \
		--arch "$ARCH" \
		-b "$BATCH" \
		-j "$WORKERS" \
		-i "$ITERS"'
