#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


if [ "$#" -lt 6 ]; then
        echo "Usage : ./run <train/eval> <num-GPU> <SRC> <DATA> <OUT> <cpu_threads> "
        exit 1
fi

SCRIPTS="../../scripts/"
OUT_DIR=$5
DATA_DIR=$4
NUM_GPU=$2
workers=$6
SRC=$3

mkdir -p $OUT_DIR
arch="transxl"
result_dir="${OUT_DIR}/${arch}_w${workers}_g${NUM_GPU}"
mkdir -p $result_dir

#dstat -cdnmgyr --output all-utils.csv 2>&1 &  
#./$SCRIPTS/free.sh &
#./$SCRIPTS/gpulog.sh &
# mpstat -P ALL 1 > cpu_util.out 2>&1 &


if [[ "$1" == 'train' ]]; then
    echo 'Run training...'
    python -m torch.distributed.launch --nproc_per_node="$NUM_GPU" $SRC/train.py \
        --config_file $SRC/wt103_base.yaml --noeval --synergy --max-iterations 50 \
        --src $SRC \
        --data $DATA_DIR \
        "${@:6}" 2>&1 > stdout.out
elif [[ "$1" == 'eval' ]]; then
    echo 'Run evaluation...'
    python -m torch.distributed.launch --nproc_per_node="$NUM_GPU" eval.py \
        --config_file wt103_base.yaml \
        "${@:6}" 2>&1 > stdout.out   
else
    echo 'unknown argment 1'
fi

#pkill -f mpstat
#pkill -f dstat
#pkill -f free
#pkill -f gpulog
#pkill -f nvidia-smi
mv *.out  $result_dir/
mv *.log $result_dir/
mv *.csv $result_dir/
