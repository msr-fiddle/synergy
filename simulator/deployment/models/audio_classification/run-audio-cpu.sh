#!/bin/bash

if [ "$#" -ne 2 ]; then
	echo "Usage : ./run-all-workers <data-dir> <out-dir>"
	exit 1
fi

DATA_DIR=$1
OUT_DIR=$2

rm train
rm val
ln -s $DATA_DIR/train train
ln -s $DATA_DIR/val val
mkdir -p $OUT_DIR

rm -rf /dev/shm/cache
mkdir /dev/shm/cache
chmod 777 /dev/shm/cache

#Max batch that fits
gpu=0
num_gpu=8
num_classes=157
echo " Data dir is $DATA_DIR"
echo " Out dir is $OUT_DIR"

#: <<'END'
for num_gpu in 8; do
	for workers in 3; do
		for batch in 128; do
			result_dir="${OUT_DIR}/audionet_b${batch}_w${workers}_g${num_gpu}"
			echo "result dir is $result_dir" 
			mkdir -p $result_dir
			echo "Now running $arch for $workers workers and $batch batch"
			mpstat -P ALL 1 > cpu_util.out 2>&1 &
			./free.sh &
			dstat -cdnmgyr --output all-utils.csv 2>&1 & 
			python -m torch.distributed.launch --nproc_per_node=$num_gpu audio-classify.py -b $batch --workers $workers --epochs 3 --dali_cpu --classes $num_classes ./ > stdout.out 2>&1
			pkill -f mpstat
			pkill -f dstat
			pkill -f free
			mv *.out  $result_dir/
			mv *.log $result_dir/
			mv *.csv $result_dir/
		done
	done
done
