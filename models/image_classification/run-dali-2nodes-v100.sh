#!/bin/bash

if [ "$#" -ne 3 ]; then
	echo "Usage : ./run-all-workers <data-dir> <out-dir> <mode=dist-mint/mint/dali>"
	exit 1
fi

DATA_DIR=$1
OUT_DIR=$2
MODE=$3
#ARCH=$4
#BATCH=$5
#CACHE=$6
#CLASSES=$7
dist_mint="dist-mint"
mint="mint"
dali="dali"

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

echo " Data dir is $DATA_DIR"
echo " Out dir is $OUT_DIR"

#: <<'END'
#for arch in 'alexnet'; do
#for arch in 'resnet50'; do
#for arch in 'resnet18'; do
#for arch in 'alexnet'  'shufflenet_v2_x0_5'; do
for arch in 'shufflenet_v2_x0_5'; do
	#for workers in 24; do
	for workers in 3; do
		#for batch in 1024 256 128 64; do
		for batch in 512; do
		#for batch in 256; do
		#for batch in 2048 512; do
		#for batch in 1024 512 256 128 64 32; do
			echo "Now running $arch for $workers workers and $batch batch"
			mpstat -P ALL 1 > cpu_util.out 2>&1 &
			./free.sh &
			dstat -cdnmgyr --output all-utils.csv 2>&1 &
			if [ "$MODE" = "$dist_mint" ]; then 
				result_dir="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}_dist_mint"
				echo "result dir is $result_dir" 
				mkdir -p $result_dir
				python -m launch --nproc_per_node=$num_gpu --nnodes=2 --node_rank=0 --master_addr="10.185.12.207" --master_port=12340 pytorch-imagenet-dali-mp.py -a $arch -b $batch --workers $workers --epochs 5 --amp --cache_size 115000 --classes 600 --dist_mint  --node_ip_list="10.185.12.207" --node_port_list 5555 --node_ip_list="10.185.12.208" --node_port_list 6666  ./ > stdout.out 2>&1
			elif [ "$MODE" = "$mint" ]; then
				result_dir="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}_mint"
				echo "result dir is $result_dir" 
				mkdir -p $result_dir
				python -m launch --nproc_per_node=$num_gpu --nnodes=2 --node_rank=0 --master_addr="10.185.12.207" --master_port=12340 pytorch-imagenet-dali-mp.py -a $arch -b $batch --workers $workers --epochs 5 --amp --cache_size 115000 --classes 600 ./ > stdout.out 2>&1
			elif [ "$MODE" = "$dali" ]; then   
				result_dir="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}_dali"
				echo "result dir is $result_dir" 
				mkdir -p $result_dir
				python -m launch --nproc_per_node=$num_gpu --nnodes=2 --node_rank=0 --master_addr="10.185.12.207" --master_port=12340 pytorch-imagenet-dali-mp.py -a $arch -b $batch --workers $workers --epochs 5 --amp --cache_size 0 --classes 600 ./ > stdout.out 2>&1
			fi
			
			pkill -f mpstat
			pkill -f dstat
			pkill -f free
			pkill -f pytorch-imagenet
			sleep 10
			mv *.out  $result_dir/
			mv *.log $result_dir/
			mv *.csv $result_dir/
		done
	done
done
exit
#END
#for arch in 'resnet18' ; do
for arch in 'resnet18'; do
	#for workers in 3; do
	for workers in 3; do
		for batch in 1024 512; do
			result_dir="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}"
			echo "result dir is $result_dir" 
			mkdir -p $result_dir
			echo "Now running $arch for $workers workers and $batch batch" 
			mpstat -P ALL 1 > cpu_util.out 2>&1 &
			./free.sh &
			dstat -cdnmgyr --output all-utils.csv 2>&1 & 
			python -m torch.distributed.launch --nproc_per_node=$num_gpu pytorch-imagenet-dali-mp.py -a $arch -b $batch --workers $workers --epochs 3 --amp ./ > stdout.out 2>&1
			pkill -f mpstat
			pkill -f dstat
			pkill -f free
			pkill -f pytorch-imagenet
			sleep 10
			mv *.out  $result_dir/
			mv *.log $result_dir/
			mv *.csv $result_dir/
		done
	done
done

#for arch in 'mobilenet_v2' ; do
for arch in 'mobilenet_v2' 'squeezenet1_0' ; do
	for workers in 3; do
	#for workers in 24; do
		for batch in 512; do
			result_dir="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}"
			echo "result dir is $result_dir" 
			mkdir -p $result_dir
			echo "Now running $arch for $workers workers and $batch batch" 
			mpstat -P ALL 1 > cpu_util.out 2>&1 &
			./free.sh &
			dstat -cdnmgyr --output all-utils.csv 2>&1 & 
			python -m torch.distributed.launch --nproc_per_node=$num_gpu pytorch-imagenet-dali-mp.py -a $arch -b $batch --workers $workers --epochs 3 --amp ./ > stdout.out 2>&1
			pkill -f mpstat
			pkill -f dstat
			pkill -f free
			pkill -f pytorch-imagenet
			sleep 10
			mv *.out  $result_dir/
			mv *.log $result_dir/
			mv *.csv $result_dir/
		done
	done
done


for arch in 'resnet50' 'vgg11' ; do
	#for workers in 24; do
	for workers in 3; do
		for batch in 512; do
			result_dir="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}"
			echo "result dir is $result_dir" 
			mkdir -p $result_dir
			echo "Now running $arch for $workers workers and $batch batch" 
			mpstat -P ALL 1 > cpu_util.out 2>&1 &
			./free.sh &
			dstat -cdnmgyr --output all-utils.csv 2>&1 & 
			python -m torch.distributed.launch --nproc_per_node=$num_gpu pytorch-imagenet-dali-mp.py -a $arch -b $batch --workers $workers --epochs 3 --dali_cpu --amp ./ > stdout.out 2>&1
			pkill -f mpstat
			pkill -f dstat
			pkill -f free
			pkill -f pytorch-imagenet
			sleep 10
			mv *.out  $result_dir/
			mv *.log $result_dir/
			mv *.csv $result_dir/
		done
	done
done

for arch in 'vgg16'; do
	for workers in 3; do
	#for workers in 3 ; do
		for batch in 256; do
			result_dir="${OUT_DIR}/${arch}_b${batch}_w${workers}_g${num_gpu}"
			echo "result dir is $result_dir" 
			mkdir -p $result_dir
			echo "Now running $arch for $workers workers and $batch batch" 
			mpstat -P ALL 1 > cpu_util.out 2>&1 &
			./free.sh &
			dstat -cdnmgyr --output all-utils.csv 2>&1 & 
			python -m torch.distributed.launch --nproc_per_node=$num_gpu pytorch-imagenet-dali-mp.py -a $arch -b $batch --workers $workers --epochs 3 --dali_cpu --amp ./ > stdout.out 2>&1
			pkill -f mpstat
			pkill -f dstat
			pkill -f free
			pkill -f pytorch-imagenet
			sleep 10
			mv *.out  $result_dir/
			mv *.log $result_dir/
			mv *.csv $result_dir/
		done
	done
done
rm train
rm val
