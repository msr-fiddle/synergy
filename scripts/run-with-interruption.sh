#!/bin/bash

if [ "$#" -ne 3 ]; then
	echo "Usage : ./run-1-job.sh <data-dir> <out-dir> <worker>"
	exit 1
fi

DATA_DIR=$1
OUT_DIR=$2
WORKER=$3
SRC="models/image_classification/"
SCRIPTS="scripts/"

rm train
rm val
#ln -s $DATA_DIR/train train
#ln -s $DATA_DIR/val val
mkdir -p $OUT_DIR


gpu=0
num_gpu=8

echo " Data dir is $DATA_DIR"
echo " Out dir is $OUT_DIR"
END=70

: <<'COMMENT'
for arch in 'resnet18'; do
	for workers in $WORKER; do
		for batch in 128; do
			echo "Now running $arch for $workers workers and $batch batch"
			result_dir_base="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}_dali_amp_cpu/synergy/"
			mpstat -P ALL 1 > cpu_util.out 2>&1 &
			./$SCRIPTS/free.sh &
			./$SCRIPTS/gpulog.sh &
			dstat -cdnmgyr --output all-utils.csv 2>&1 &
 			for i in $(seq 1 $END); do
				result_dir="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}_dali_amp_cpu/synergy/un${i}"
				echo "result dir is $result_dir" 
				mkdir -p $result_dir
				python -m torch.distributed.launch --nproc_per_node=$num_gpu $SRC/pytorch-imagenet.py -a $arch -b $batch --workers $workers --epochs 70 --amp --dali --dali_cpu  --synergy --data $DATA_DIR > stdout.out 2>&1
				sleep 2
				pkill -f pytorch-imagenet
				mv stdout.out  $result_dir/
				mv *.log $result_dir/
				mv acc-*.csv $result_dir/
				cp chk/* $result_dir/
			done
			echo "RAN $arch for $workers workers, $batch batch with DALI CPU" >> stdout.out
			pkill -f mpstat
			pkill -f dstat
			pkill -f free
			pkill -f gpulog
			pkill -f nvidia-smi
			mv *.out  $result_dir_base/
			mv *.log $result_dir_base/
			mv *.csv $result_dir_base/
		done
	done
done
exit
COMMENT


for arch in 'resnet18'; do
	for workers in $WORKER; do
		for batch in 128; do
			echo "Now running $arch for $workers workers and $batch batch"
			result_dir_base="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}_dali_amp_cpu/orig/"
			mpstat -P ALL 1 > cpu_util.out 2>&1 &
			./$SCRIPTS/free.sh &
			./$SCRIPTS/gpulog.sh &
			dstat -cdnmgyr --output all-utils.csv 2>&1 &
 			for i in $(seq 1 $END); do
				result_dir="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}_dali_amp_cpu/orig/run${i}"
				echo "result dir is $result_dir" 
				mkdir -p $result_dir
				python -m torch.distributed.launch --nproc_per_node=$num_gpu $SRC/pytorch-imagenet.py -a $arch -b $batch --workers $workers --epochs 70 --amp --dali --dali_cpu  --step-lr --data $DATA_DIR > stdout.out 2>&1
				#python -m torch.distributed.launch --nproc_per_node=$num_gpu $SRC/pytorch-imagenet.py -a $arch -b $batch --workers $workers --epochs 70 --amp --dali --dali_cpu  --data $DATA_DIR > stdout.out 2>&1
				sleep 2
				pkill -f pytorch-imagenet
				mv stdout.out  $result_dir/
				mv *.log $result_dir/
				mv acc-*.csv $result_dir/
				cp chk/* $result_dir/
			done
			echo "RAN $arch for $workers workers, $batch batch with DALI CPU" >> stdout.out
			pkill -f mpstat
			pkill -f dstat
			pkill -f free
			pkill -f gpulog
			pkill -f nvidia-smi
			mv *.out  $result_dir_base/
			mv *.log $result_dir_base/
			mv *.csv $result_dir_base/
		done
	done
done



rm train
rm val
