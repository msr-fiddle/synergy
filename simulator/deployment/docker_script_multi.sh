#!/bin/bash

if [ "$#" -ne 5 ]; then
    echo "Usage : ./run.sh <out-dir> <sched_ip> <sched_port> <job_id> <chk dir>"
    exit 1
fi

OUT_DIR=$1
SCHED_IP=$2
SCHED_PORT=$3
JOB_ID=$4
CHK_DIR=$5

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
ELAPSED_ITERS=${SYNERGY_ELAPSED_ITERS%.*}

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

echo $RANK, $GPUS, $CPUS_PER_GPU, $MEM, $ITERS, $ARCH, $BATCH, $GPU_IDS, $CPU_IDS, $LOG_DIR, $CHK_DIR

if [ "$ARCH" = "gnmt" ]; then

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
	-e "CHK_DIR="$CHK_DIR"" \
	-e "CPUS_PER_GPU="$CPUS_PER_GPU"" \
	-e "GPUS="$GPUS"" \
	-e "GPU_IDS="$GPU_IDS"" \
  -e "ELAPSED_ITERS="$ELAPSED_ITERS"" \
	fiddlev3.azurecr.io/synergy_dali:latest /bin/bash \
	-c 'cd /datadrive/mnt2/jaya/scheduler-sim/python-simulator/deployment; \
	CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$GPUS  ./models/GNMT/train.py \
    --sched-addr "$SCHED_IP" \
    --sched-port "$SCHED_PORT" \
    --job-id "$JOB_ID" \
    --log-dir "$LOG_DIR" \
    -b "$BATCH" \
    -j "$CPUS_PER_GPU" \
    --chk_dir "$CHK_DIR" \
    --dataset-dir "/datadrive/mnt2/jaya/datasets/wmt_ende/" \
    --synergy \
    --elapsed_iters "$ELAPSED_ITERS" \
    --max-iterations "$ITERS"'

elif [ "$ARCH" = "transformer" ]; then

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
	-e "CHK_DIR="$CHK_DIR"" \
	-e "CPUS_PER_GPU="$CPUS_PER_GPU"" \
	-e "GPUS="$GPUS"" \
	-e "GPU_IDS="$GPU_IDS"" \
  -e "ELAPSED_ITERS="$ELAPSED_ITERS"" \
	fiddlev3.azurecr.io/synergy_dali:latest /bin/bash \
	-c 'cd /datadrive/mnt2/jaya/scheduler-sim/python-simulator/deployment; \
	CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$GPUS  ./models/Transformer-XL/pytorch/train.py \
    --sched-addr "$SCHED_IP" \
    --sched-port "$SCHED_PORT" \
    --job-id "$JOB_ID" \
    --log-dir "$LOG_DIR" \
    -b "$BATCH" \
    --config_file "./models/Transformer-XL/pytorch/wt103_base.yaml" \
    --src "./models/Transformer-XL/pytorch/" \
    -j "$CPUS_PER_GPU" \
    --chk_dir "$CHK_DIR" \
    --data "/datadrive/mnt2/jaya/datasets/wikitext-103/" \
    --synergy \
    --noeval \
    --elapsed_iters "$ELAPSED_ITERS" \
    --max-iterations "$ITERS"'


elif [ "$ARCH" = "m5" ]; then

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
	-e "CHK_DIR="$CHK_DIR"" \
	-e "CPUS_PER_GPU="$CPUS_PER_GPU"" \
	-e "GPUS="$GPUS"" \
	-e "GPU_IDS="$GPU_IDS"" \
  -e "ELAPSED_ITERS="$ELAPSED_ITERS"" \
	fiddlev3.azurecr.io/synergy_dali:latest /bin/bash \
	-c 'cd /datadrive/mnt2/jaya/scheduler-sim/python-simulator/deployment; \
	CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$GPUS  ./models/audio_classification/audio-classify.py \
    --sched-addr "$SCHED_IP" \
    --sched-port "$SCHED_PORT" \
    --job-id "$JOB_ID" \
    --log-dir "$LOG_DIR" \
    -b "$BATCH" \
    --classes 19 \
    --workers "$CPUS_PER_GPU" \
    --amp \
    --dali_cpu \
    --chk_dir "$CHK_DIR" \
    --data "/datadrive/mnt3/jaya/datasets/fma_wav_small/" \
    --synergy \
    --noeval \
    --elapsed_iters "$ELAPSED_ITERS" \
    --max-iterations "$ITERS"'



elif [ "$ARCH" = "deepspeech" ]; then

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
	-e "CHK_DIR="$CHK_DIR"" \
	-e "CPUS_PER_GPU="$CPUS_PER_GPU"" \
	-e "GPUS="$GPUS"" \
	-e "GPU_IDS="$GPU_IDS"" \
  -e "ELAPSED_ITERS="$ELAPSED_ITERS"" \
	fiddlev3.azurecr.io/synergy_dali:latest /bin/bash \
	-c 'cd /datadrive/mnt2/jaya/scheduler-sim/python-simulator/deployment; \
	CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$GPUS  ./models/deepspeech.pytorch/train.py \
    --sched-addr "$SCHED_IP" \
    --sched-port "$SCHED_PORT" \
    --job-id "$JOB_ID" \
    --log-dir "$LOG_DIR" \
    -b "$BATCH" \
    -j "$CPUS_PER_GPU" \
    --chk_dir "$CHK_DIR" \
    --train-manifest "./models/deepspeech.pytorch/data/libri_train_manifest.csv" \
    --val-manifest "./models/deepspeech.pytorch/data//libri_val_manifest.csv" \
    --labels-path "./models/deepspeech.pytorch/labels.json" \
    --noeval \
    --synergy \
    --no-sortaGrad \
    --opt-level 'O1' \
    --loss-scale 1 \
    --rnn-type lstm \
    --hidden-size 1024 \
    --hidden-layers 5 \
    --rnn-type lstm \
    --id libri \
    --cuda \
    --learning-anneal 1.01 \
    --elapsed_iters "$ELAPSED_ITERS" \
    --max-iterations "$ITERS"'


elif [ "$ARCH" = "lstm" ]; then

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
	-e "CHK_DIR="$CHK_DIR"" \
	-e "CPUS_PER_GPU="$CPUS_PER_GPU"" \
	-e "GPUS="$GPUS"" \
	-e "GPU_IDS="$GPU_IDS"" \
  -e "ELAPSED_ITERS="$ELAPSED_ITERS"" \
	fiddlev3.azurecr.io/synergy_dali:latest /bin/bash \
	-c 'cd /datadrive/mnt2/jaya/scheduler-sim/python-simulator/deployment; \
	CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$GPUS  ./models/LSTM/main.py \
    --sched-addr "$SCHED_IP" \
    --sched-port "$SCHED_PORT" \
    --job-id "$JOB_ID" \
    --log-dir "$LOG_DIR" \
    -b "$BATCH" \
    -j "$CPUS_PER_GPU" \
    --chk_dir "$CHK_DIR" \
    --data "/datadrive/mnt2/jaya/datasets/wikitext-2/" \
    --noeval \
    --synergy \
    --cuda \
    --elapsed_iters "$ELAPSED_ITERS" \
    --max-iterations "$ITERS"'


else
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
	-e "CHK_DIR="$CHK_DIR"" \
	-e "CPUS_PER_GPU="$CPUS_PER_GPU"" \
	-e "GPUS="$GPUS"" \
	-e "GPU_IDS="$GPU_IDS"" \
  -e "ELAPSED_ITERS="$ELAPSED_ITERS"" \
	nvidia/dali:py36_cu10.run /bin/bash \
	-c 'cd /datadrive/mnt2/jaya/scheduler-sim/python-simulator/deployment; \
	CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$GPUS pytorch-imagenet.py \
    --sched-addr "$SCHED_IP" \
    --sched-port "$SCHED_PORT" \
    --job-id "$JOB_ID" \
    --arch "$ARCH" \
    --log-dir "$LOG_DIR" \
    -b "$BATCH" \
    -j "$CPUS_PER_GPU" \
    --chk_dir "$CHK_DIR" \
    --data "/datadrive/mnt2/jaya/datasets/imagenet/" \
    --synergy \
    --dali \
    --noeval \
    --amp \
    --dali_cpu \
    --elapsed_iters "$ELAPSED_ITERS" \
    --max-iterations "$ITERS"'
fi
