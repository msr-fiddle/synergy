# limitations under the License.
if [ "$#" -ne 2 ]; then    
	echo "Usage : ./run-all-workers output_dir worker_per_gpu "
	exit 1
fi

OUTPUT_DIR=$1
workers=$2
num_gpus=1
mkdir -p $OUTPUT_DIR

SCRIPTS="../scripts"


result_dir="${OUTPUT_DIR}/LSTM_w${workers}_g${num_gpus}"

mkdir -p $result_dir
echo "Result dir is $result_dir, worker per gpu= $workers, gpu= $num_gpus"


dstat -cdnmgyr --output all-utils.csv 2>&1 &
./$SCRIPTS/gpulog.sh &  
mpstat -P ALL 1 > cpu_util.out 2>&1 &  
./$SCRIPTS/free.sh & 
python main.py --cuda --epochs 2 2>&1 > stdout.out

pkill -f dstat
pkill -f mpstat
pkill -f free    
pkill -f gpulog 
pkill -f nvidia-smi   
pkill -f run_pretraining.sh
sleep 3
mv *.log $result_dir/
mv *.csv $result_dir/
mv *.out $result_dir/



