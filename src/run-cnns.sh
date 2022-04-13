#!/bin/bash

#if [ "$#" -ne 3 ]; then
#	echo "Usage : ./run-1-job.sh <data-dir> <out-dir> <worker>"
#	exit 1
#fi

#DATA_DIR=$1
#OUT_DIR=$2
WORKER=3



echo " Data dir is $DATA_DIR"
echo " Out dir is $OUT_DIR"


#for arch in 'shufflenet_v2_x0_5' 'mobilenet_v2' 'squeezenet1_0' 'resnet50' 'vgg11' 'resnet18'; do
for arch in 'alexnet' ; do
	for workers in $WORKER; do
		for batch in 512; do
      job_name="${arch}_512_1_cpu_imagenet"
			echo "Now running $arch for $workers workers and $batch batch"
      start=$SECONDS
      python profiler/offline_profiler.py --job-name $job_name --cpu 24 --num-gpus 1 -b $batch --docker-img=nvcr.io/nvidia/pytorch:21.11-py3 ../models/image_classification/pytorch-imagenet.py -a $arch --dali --amp --dali_cpu --max-iterations 50 --synergy --noeval --workers 3 -b $batch --data '/datadrive/home/amar/dataset/imagenet/'
      #python profiler/offline_profiler.py --job-name $job_name --cpu 24 --num-gpus 1 -b $batch --docker-img=nvidia/dali:py36_cu10.run ../models/image_classification/pytorch-imagenet.py -a $arch --dali --amp --dali_cpu --max-iterations 50 --synergy --noeval --workers 3 -b $batch --data '/datadrive/home/amar/dataset/imagenet/'
      duration=$(( SECONDS - start ))
			echo "RAN $arch for $workers workers, $batch for $duration s"
			sleep 2



: <<'END'

for arch in 'alexnet' 'shufflenet_v2_x0_5' 'mobilenet_v2' 'squeezenet1_0' 'resnet50' 'vgg11' 'resnet18'; do
	for workers in $WORKER; do
		for batch in 512; do
      job_name="${arch}_gpu_openimages"
			echo "Now running $arch for $workers workers and $batch batch"
      start=$SECONDS
      python profiler/offline_profiler.py --job-name $job_name --cpu 24 --num-gpus 1 -b $batch --docker-img=nvidia/dali:py36_cu10.run ../models/image_classification/pytorch-imagenet.py -a $arch --dali --amp --max-iterations 50 --workers 3 -b $batch --data '/datadrive/mnt4/jaya/datasets/openimages/'
      duration=$(( SECONDS - start ))
			echo "RAN $arch for $workers workers, $batch for $duration s"
			sleep 2


      job_name="${arch}_cpu_openimages"
			echo "Now running $arch for $workers workers and $batch batch"
      start=$SECONDS
      python profiler/offline_profiler.py --job-name $job_name --cpu 24 --num-gpus 1 -b $batch --docker-img=nvidia/dali:py36_cu10.run ../models/image_classification/pytorch-imagenet.py -a $arch --dali --amp --dali_cpu --max-iterations 50 --workers 3 -b $batch --data '/datadrive/mnt4/jaya/datasets/openimages/'
      duration=$(( SECONDS - start ))
			echo "RAN $arch for $workers workers, $batch for $duration s"
			sleep 2

      job_name="${arch}_gpu_imagenet"
			echo "Now running $arch for $workers workers and $batch batch"
      start=$SECONDS
      python profiler/offline_profiler.py --job-name $job_name --cpu 24 --num-gpus 1 -b $batch --docker-img=nvidia/dali:py36_cu10.run ../models/image_classification/pytorch-imagenet.py -a $arch --dali --amp --max-iterations 50 --workers 3 -b $batch --data '/datadrive/mnt4/jaya/datasets/openimages/'
      duration=$(( SECONDS - start ))
			echo "RAN $arch for $workers workers, $batch for $duration s"
			sleep 2


      job_name="${arch}_cpu_imagenet"
			echo "Now running $arch for $workers workers and $batch batch"
      start=$SECONDS
      python profiler/offline_profiler.py --job-name $job_name --cpu 24 --num-gpus 1 -b $batch --docker-img=nvidia/dali:py36_cu10.run ../models/image_classification/pytorch-imagenet.py -a $arch --dali --amp --max-iterations 50 --workers 3 -b $batch --data '/datadrive/mnt4/jaya/datasets/openimages/'
      duration=$(( SECONDS - start ))
			echo "RAN $arch for $workers workers, $batch for $duration s"
			sleep 2
END

		done
	done
done
