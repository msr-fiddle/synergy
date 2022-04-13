#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage : ./run.sh <out-dir> <sched_ip> <sched_port> <job_id>"
    exit 1
fi

OUT_DIR=$1/50g/
SCHED_IP=$2
SCHED_PORT=$3
JOB_ID=$4

TOTAL_CPU=${SYNERGY_CPU_THIS_SERVER%.*}
MEM=${SYNERGY_MEM_THIS_SERVER%.*}
GPUS=${SYNERGY_GPU_THIS_SERVER%.*}
CPUS_PER_GPU_FLOAT=$((TOTAL_CPU / GPUS))
CPUS_PER_GPU=${CPUS_PER_GPU_FLOAT%.*}
ITERS=${SYNERGY_TOTAL_ITERS%.*}
ARCH=$SYNERGY_ARCH
BATCH=${SYNERGY_BATCH_SIZE%.*}
RANK=${SYNERGY_SERVER_RANK%.*}
GPU_IDS=${SYNERGY_GPUS_ALLOCATED%.*}
CPU_IDS=${SYNERGY_CPUS_ALLOCATED%.*}
LOG_DIR=$SYNERGY_LOG_DIR


if [ "$ARCH" = "res18" ]; then
    ARCH="resnet18"
elif [ "$ARCH" = "res50" ]; then
    ARCH="resnet50"
elif [ "$ARCH" = "mobilenet" ]; then
    ARCH="mobilenet_v2"
elif [ "$ARCH" = "shufflenet" ]; then
    ARCH="shufflenet_v2_x0_5"
elif [ "$ARCH" = "squeezenet" ]; then
    ARCH="squeezenet1_0"
elif [ "$ARCH" = "vgg" ]; then
    ARCH="vgg11"
fi

echo $RANK, $GPUS, $CPUS_PER_GPU, $MEM, $ITERS, $ARCH, $BATCH, $GPU_IDS, $CPU_IDS, $LOG_DIR

#--cpus="$TOTAL_CPU" --cpuset-cpus=0-$(($WORKERS-1)) \

nvidia-docker run \
	--ipc=host \
	--mount src=/,target=/datadrive/,type=bind --rm -it \
	--network=host \
	--cpus="$TOTAL_CPU" --cpuset-cpus=$CPU_IDS \
	-m "$MEM"g \
	--privileged \
	-e "SCHED_PORT="$SCHED_PORT"" \
	-e "SCHED_IP="$SCHED_IP"" \
	-e "JOB_ID="$JOB_ID""  \
	-e "ITERS="$ITERS"" \
	-e "ARCH="$ARCH"" \
	-e "BATCH="$BATCH"" \
	-e "LOG_DIR="$LOG_DIR"" \
	-e "CPUS_PER_GPU="$CPUS_PER_GPU"" \
	-e "GPUS="$GPUS"" \
	-e "GPU_IDS="$GPU_IDS"" \
	nvidia/dali:py36_cu10.run /bin/bash \
	-c 'cd /datadrive/mnt2/jaya/scheduler-sim/python-simulator/deployment; \
	python job_script_multi.py \
		--sched-addr "$SCHED_IP" \
		--sched-port "$SCHED_PORT" \
		--job-id "$JOB_ID" \
		--arch "$ARCH" \
		--log-dir "$LOG_DIR" \
		-b "$BATCH" \
		-j "$CPUS_PER_GPU" \
    --data "/datadrive/mnt2/jaya/datasets/imagenet/" \
		--max-iterations "$ITERS"'
#--gpus "$GPUS" \
#-i "$ITERS"'
#python job_script_multi.py \
