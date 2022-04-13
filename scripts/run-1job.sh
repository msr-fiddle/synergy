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

: <<'END'
#for arch in 'alexnet' 'shufflenet_v2_x0_5' 'mobilenet_v2' 'squeezenet1_0' 'resnet50' 'vgg11'; do
for arch in 'resnet18'; do
	for workers in $WORKER; do
		for batch in 512; do
			result_dir="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}_dali_amp"
			echo "result dir is $result_dir" 
			mkdir -p $result_dir
			echo "Now running $arch for $workers workers and $batch batch"
			mpstat -P ALL 1 > cpu_util.out 2>&1 &
			./$SCRIPTS/free.sh &
			./$SCRIPTS/gpulog.sh &
			dstat -cdnmgyr --output all-utils.csv 2>&1 & 
			CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$num_gpu $SRC/pytorch-imagenet.py -a $arch -b $batch --workers $workers --epochs 3 --amp --dali ./ > stdout.out 2>&1
			echo "RAN $arch for $workers workers, $batch batch with DALI GPU mixed" >> stdout.out
			pkill -f mpstat
			pkill -f dstat
			pkill -f free
			pkill -f gpulog
			pkill -f nvidia-smi
			pkill -f pytorch-imagenet
			sleep 2
			mv *.out  $result_dir/
			mv *.log $result_dir/
			mv *.csv $result_dir/
		done
	done
done
for arch in 'resnet18'; do
	for workers in $WORKER; do
		for batch in 512; do
			result_dir="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}_pytorch_amp"
			echo "result dir is $result_dir" 
			mkdir -p $result_dir
			echo "Now running $arch for $workers workers and $batch batch"
			mpstat -P ALL 1 > cpu_util.out 2>&1 &
			./$SCRIPTS/free.sh &
			./$SCRIPTS/gpulog.sh &
			dstat -cdnmgyr --output all-utils.csv 2>&1 & 
			python -m torch.distributed.launch --nproc_per_node=$num_gpu $SRC/pytorch-imagenet.py -a $arch -b $batch --workers $workers --epochs 1 --amp --max-iterations 50 --noeval --data $DATA_DIR > stdout.out 2>&1
			echo "RAN $arch for $workers workers, $batch batch with DDP" >> stdout.out
			pkill -f mpstat
			pkill -f dstat
			pkill -f free
			pkill -f gpulog
			pkill -f nvidia-smi
			pkill -f pytorch-imagenet
			sleep 2
			mv *.out  $result_dir/
			mv *.log $result_dir/
			mv *.csv $result_dir/
		done
	done
done

exit
#END
END

for arch in 'resnet18'; do
	for workers in $WORKER; do
		for batch in 512; do
			result_dir="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}_dali_amp_cpu"
			echo "result dir is $result_dir" 
			mkdir -p $result_dir
			echo "Now running $arch for $workers workers and $batch batch"
			mpstat -P ALL 1 > cpu_util.out 2>&1 &
			./$SCRIPTS/free.sh &
			./$SCRIPTS/gpulog.sh &
			dstat -cdnmgyr --output all-utils.csv 2>&1 & 
			python -m torch.distributed.launch --nproc_per_node=$num_gpu $SRC/pytorch-imagenet.py -a $arch -b $batch --workers $workers --epochs 8 --amp --dali --dali_cpu  --noeval --data $DATA_DIR > stdout.out 2>&1
			#python -m torch.distributed.launch --nproc_per_node=$num_gpu $SRC/pytorch-imagenet.py -a $arch -b $batch --workers $workers --epochs 1 --amp --dali --dali_cpu  --max-iterations 50 --noeval --data $DATA_DIR > stdout.out 2>&1
			echo "RAN $arch for $workers workers, $batch batch with DALI CPU" >> stdout.out
			pkill -f mpstat
			pkill -f dstat
			pkill -f free
			pkill -f gpulog
			pkill -f nvidia-smi
			pkill -f pytorch-imagenet
			sleep 2
			mv *.out  $result_dir/
			mv *.log $result_dir/
			mv *.csv $result_dir/
			cp chk/* $result_dir/
		done
	done
done

rm train
rm val
