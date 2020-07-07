#!/bin/bash

if [ "$#" -ne 3 ]; then
	echo "Usage : ./run-1-job.sh <data-dir> <out-dir> <worker>"
	exit 1
fi

DATA_DIR=$1
OUT_DIR=$2
WORKER=$3
SRC="image_classification/"
SCRIPTS="scripts/"

mkdir -p $OUT_DIR
rm -rf /dev/shm/cache
mkdir /dev/shm/cache
chmod 777 /dev/shm/cache

gpu=0
num_gpu=4

echo " Data dir is $DATA_DIR"
echo " Out dir is $OUT_DIR"


for arch in 'resnet18'; do
	for workers in $WORKER; do
		for batch in 1024; do
#			result_dir="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}_dali_amp_gpu"
			result_dir="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}_dali_amp_cpu"
			echo "result dir is $result_dir" 
			mkdir -p $result_dir
			echo "Now running $arch for $workers workers and $batch batch"
			#mpstat -P ALL 1 > cpu_util.out 2>&1 &
			#$SCRIPTS/free.sh &
			#$SCRIPTS/gpulog.sh &
			#dstat -cdnmgyr --output all-utils.csv 2>&1 & 
			CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$num_gpu --master_port=29600 $SRC/pytorch-imagenet.py -a $arch -b $batch --workers $workers --epochs 2 --amp --dali --dali_cpu  $DATA_DIR > stdout_1.out 2>&1
#380000
			#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$num_gpu --master_port=29600 $SRC/pytorch-imagenet.py -a $arch -b $batch --workers $workers --epochs 2 --amp --dali --dali_cpu --classes 4260 $DATA_DIR > stdout_1.out 2>&1
			echo "RAN $arch for $workers workers, $batch batch with DALI CPU" >> stdout_1.out
			#pkill -f mpstat
			#pkill -f dstat
			#pkill -f free
			#pkill -f gpulog
			#pkill -f nvidia-smi
			#pkill -f pytorch-imagenet
			sleep 2
			mv stdout_1.out  $result_dir/
			#mv *.log $result_dir/
			#mv *.csv $result_dir/
		done
	done
done

