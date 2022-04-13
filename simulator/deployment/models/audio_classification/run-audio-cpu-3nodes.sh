#!/bin/bash

if [ "$#" -ne 3 ]; then
	echo "Usage : ./run-all-workers <data-dir> <out-dir>"
	exit 1
fi

DATA_DIR=$1
OUT_DIR=$2
MODE=$3
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
cp ../launch.py ./   

#Max batch that fits
gpu=0
num_gpu=8

echo " Data dir is $DATA_DIR"
echo " Out dir is $OUT_DIR"

for num_gpu in 8; do   
	for workers in 3; do
		for batch in 128; do
			echo "Now running $arch for $workers workers and $batch batch"
			mpstat -P ALL 1 > cpu_util.out 2>&1 &
			./free.sh &
			dstat -cdnmgyr --output all-utils.csv 2>&1 & 
			if [ "$MODE" = "$dist_mint" ]; then   			
				result_dir="${OUT_DIR}/audionet_b${batch}_w${workers}_g${num_gpu}_dist_mint"
				echo "result dir is $result_dir" 
				mkdir -p $result_dir
				python -m launch --nproc_per_node=$num_gpu --nnodes=2 --node_rank=0 --master_addr="10.185.12.207" --master_port=12340 audio-classify.py -b $batch --workers $workers --epochs 10 --cache_size 5500 --dist_mint --node_ip_list="10.185.12.207" --node_port_list 5555 --node_ip_list="10.185.12.208" --node_port_list 6666 ./ > stdout.out 2>&1
			elif [ "$MODE" = "$mint" ]; then  
				result_dir="${OUT_DIR}/audionet_b${batch}_w${workers}_g${num_gpu}_mint"
				echo "result dir is $result_dir" 
				mkdir -p $result_dir
				python -m launch --nproc_per_node=$num_gpu --nnodes=2 --node_rank=0 --master_addr="10.185.12.207" --master_port=12340 audio-classify.py -b $batch --workers $workers --epochs 10 --cache_size 5500 ./ > stdout.out 2>&1
			elif [ "$MODE" = "$dali" ]; then 
				result_dir="${OUT_DIR}/audionet_b${batch}_w${workers}_g${num_gpu}_dali"
				echo "result dir is $result_dir" 
				mkdir -p $result_dir
				python -m launch --nproc_per_node=$num_gpu --nnodes=2 --node_rank=0 --master_addr="10.185.12.207" --master_port=12340 audio-classify.py -b $batch --workers $workers --epochs 10 --cache_size 0 ./ > stdout.out 2>&1
			fi
			pkill -f mpstat
			pkill -f dstat
			pkill -f free
			pkill -f audio-classify
			sleep 5
			mv *.out  $result_dir/
			mv *.log $result_dir/
			mv *.csv $result_dir/
		done
	done
done
