#!/bin/bash

if [ "$#" -ne 2 ]; then
	echo "Usage : ./run-1-job.sh <data-dir> <out-dir>"
	exit 1
fi

DATA_DIR=$1
OUT_DIR=$2

rm train
rm val
ln -s $DATA_DIR/train train
ln -s $DATA_DIR/val val
mkdir -p $OUT_DIR


num_gpu=2

echo " Data dir is $DATA_DIR"
echo " Out dir is $OUT_DIR"

#for arch in 'alexnet' 'shufflenet_v2_x0_5' 'mobilenet_v2' 'squeezenet1_0' 'resnet50' 'vgg11'; do
for arch in 'resnet18'; do
	for workers in 3; do
		for batch in 512; do
			result_dir="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}_dali_amp"
			echo "result dir is $result_dir" 
			mkdir -p $result_dir
			echo "Now running $arch for $workers workers and $batch batch"
			mpstat -P ALL 1 > cpu_util.out 2>&1 &
			./free.sh &
			dstat -cdnmgyr --output all-utils.csv 2>&1 & 
			CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$num_gpu pytorch-imagenet.py -a $arch -b $batch --workers $workers --epochs 3 --amp --dali ./ > stdout.out 2>&1
			echo "RAN $arch for $workers workers, $batch batch with DALI GPU mixed" >> stdout.out
			pkill -f mpstat
			pkill -f dstat
			pkill -f free
			pkill -f pytorch-imagenet
			sleep 2
			mv *.out  $result_dir/
			mv *.log $result_dir/
			mv *.csv $result_dir/
		done
	done
done

for arch in 'resnet18'; do
	for workers in 3; do
		for batch in 512; do
			result_dir="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}_pytorch_amp"
			echo "result dir is $result_dir" 
			mkdir -p $result_dir
			echo "Now running $arch for $workers workers and $batch batch"
			mpstat -P ALL 1 > cpu_util.out 2>&1 &
			./free.sh &
			dstat -cdnmgyr --output all-utils.csv 2>&1 & 
			CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$num_gpu pytorch-imagenet.py -a $arch -b $batch --workers $workers --epochs 3 --amp ./ > stdout.out 2>&1
			echo "RAN $arch for $workers workers, $batch batch with DDP" >> stdout.out
			pkill -f mpstat
			pkill -f dstat
			pkill -f free
			pkill -f pytorch-imagenet
			sleep 2
			mv *.out  $result_dir/
			mv *.log $result_dir/
			mv *.csv $result_dir/
		done
	done
done


for arch in 'resnet18'; do
	for workers in 3; do
		for batch in 512; do
			result_dir="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}_dali_amp_cpu"
			echo "result dir is $result_dir" 
			mkdir -p $result_dir
			echo "Now running $arch for $workers workers and $batch batch"
			mpstat -P ALL 1 > cpu_util.out 2>&1 &
			./free.sh &
			dstat -cdnmgyr --output all-utils.csv 2>&1 & 
			CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$num_gpu pytorch-imagenet.py -a $arch -b $batch --workers $workers --epochs 3 --amp --dali --dali_cpu ./ > stdout.out 2>&1
			echo "RAN $arch for $workers workers, $batch batch with DALI CPU" >> stdout.out
			pkill -f mpstat
			pkill -f dstat
			pkill -f free
			pkill -f pytorch-imagenet
			sleep 2
			mv *.out  $result_dir/
			mv *.log $result_dir/
			mv *.csv $result_dir/
		done
	done
done

rm train
rm val
