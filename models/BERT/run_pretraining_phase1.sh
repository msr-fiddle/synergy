#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
if [ "$#" -ne 3 ]; then    
	echo "Usage : ./run-all-workers data_dir output_dir worker_per_gpu "
	exit 1
fi

BERT_PREP_WORKING_DIR=$1
OUTPUT_DIR=$2

worker_per_gpu=$3
mkdir -p $OUTPUT_DIR

SCRIPTS="../scripts"
echo "Container nvidia build = " $NVIDIA_BUILD_ID
#Per GPU batch size
train_batch_size=1024
#train_batch_size=8192
learning_rate="6e-3"
precision="fp16"
num_gpus=1
#num_gpus=8
warmup_proportion="0.2843"
train_steps=30
#train_steps=7038
save_checkpoint_steps=${7:-200}
resume_training=${8:-"false"}
create_logfile=${9:-"true"}
accumulate_gradients=${10:-"true"}
gradient_accumulation_steps=${11:-128}
seed=${12:-$RANDOM}
job_name=${13:-"bert_lamb_pretraining"}
allreduce_post_accumulation=${14:-"true"}
allreduce_post_accumulation_fp16=${15:-"true"}
accumulate_into_fp16=${16:-"false"}
workers=$worker_per_gpu


result_dir="${OUTPUT_DIR}/${job_name}_phase1_b${train_batch_size}_w${workers}_g${num_gpus}"

result1="${result_dir}/phase1"
rm -rf $result1
mkdir -p $result1
echo "Result dir is $result1, worker per gpu= $workers, gpu= $num_gpus"

DATASET=hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus/train # change this for other datasets
DATA_DIR=$BERT_PREP_WORKING_DIR/${DATASET}/
BERT_CONFIG=bert_config.json
RESULTS_DIR=/workspace/bert/results
CHECKPOINTS_DIR=$RESULTS_DIR/checkpoints

rm -rf $CHECKPOINTS_DIR

mkdir -p $CHECKPOINTS_DIR


if [ ! -d "$DATA_DIR" ] ; then
   echo "Warning! $DATA_DIR directory missing. Training cannot start"
fi
if [ ! -d "$RESULTS_DIR" ] ; then
   echo "Error! $RESULTS_DIR directory missing."
   exit -1
fi
if [ ! -d "$CHECKPOINTS_DIR" ] ; then
   echo "Warning! $CHECKPOINTS_DIR directory missing."
   echo "Checkpoints will be written to $RESULTS_DIR instead."
   CHECKPOINTS_DIR=$RESULTS_DIR
fi
if [ ! -f "$BERT_CONFIG" ] ; then
   echo "Error! BERT large configuration file not found at $BERT_CONFIG"
   exit -1
fi

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ] ; then
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps"
fi

CHECKPOINT=""
if [ "$resume_training" == "true" ] ; then
   CHECKPOINT="--resume_from_checkpoint"
fi

ALL_REDUCE_POST_ACCUMULATION=""
if [ "$allreduce_post_accumulation" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION="--allreduce_post_accumulation"
fi

ALL_REDUCE_POST_ACCUMULATION_FP16=""
if [ "$allreduce_post_accumulation_fp16" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION_FP16="--allreduce_post_accumulation_fp16"
fi

ACCUMULATE_INTO_FP16=""
if [ "$accumulate_into_fp16" == "true" ] ; then
   ACCUMULATE_INTO_FP16="--accumulate_into_fp16"
fi

echo $DATA_DIR
INPUT_DIR=$DATA_DIR
CMD="run_pretraining.py"
CMD+=" --input_dir=$DATA_DIR"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --bert_model=bert-large-uncased"
CMD+=" --train_batch_size=$train_batch_size"
CMD+=" --max_seq_length=128"
CMD+=" --max_predictions_per_seq=20"
CMD+=" --max_steps=$train_steps"
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate"
CMD+=" --seed=$seed"
CMD+=" --workers=$workers"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" --do_train"

#if [ "$num_gpus" -gt 1  ] ; then
CMD="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus $CMD"
#else
#   CMD="python3  $CMD"
#fi

dstat -cdnmgyr --output all-utils.csv 2>&1 &
./$SCRIPTS/gpulog.sh &  
mpstat -P ALL 1 > cpu_util.out 2>&1 &  
./$SCRIPTS/free.sh & 

if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size \* $num_gpus)
  printf -v TAG "pyt_bert_pretraining_phase1_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$result1/$job_name.$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD 
else
   (
     $CMD
   ) 2>&1 > $LOGFILE
   #) |& tee $LOGFILE
fi

set +x

pkill -f dstat
pkill -f mpstat
pkill -f free    
pkill -f gpulog 
pkill -f nvidia-smi   
pkill -f run_pretraining.sh
sleep 3
mv *.log $result1/
mv *.csv $result1/
mv *.out $result1/

echo "finished pretraining, starting benchmarking"

target_loss=15
THROUGHPUT=10
THRESHOLD=0.9

throughput=`cat $LOGFILE | grep Iteration | tail -1 | awk -F'it/s' '{print $1}' | awk -F',' '{print $2}' | egrep -o [0-9.]+`
loss=`cat $LOGFILE | grep 'Average Loss' | tail -1 | awk -F'Average Loss =' '{print $2}' | awk -F' ' '{print $1}' | egrep -o [0-9.]+`
final_loss=`cat $LOGFILE | grep 'Total Steps' | tail -1 | awk -F'Final Loss =' '{print $2}' | awk -F' ' '{print $1}' | egrep -o [0-9.]+`

train_perf=$(awk 'BEGIN {print ('$throughput' * '$num_gpus' * '$train_batch_size')}')
echo " training throughput phase1: $train_perf sequences/second"
echo "average loss: $loss"
echo "final loss: $final_loss"


