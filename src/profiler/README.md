## TODO

1. Improve the CPU datapoint search algo :  Currently it is:
  ```
  cpu = MAX_CPU
  prev_perf = -1
  while cpu > 0:
    perf = run(cpu)
    if within10Percent(perf, prev_perf) and prev_perf >= 0:
      cpu = cpu/2     # take a big step 
    else
      cpu = cpu - 1   # take a small step
  ```
  
  2. Haven't tested profiling in multi-server settings
  3. Compare profiling results with empirical runs for different models
  4. Fix 7-10 different models and get their profiles to be used by the scheduler-simulator
  5. Export profiler output as json. Currently simply prints out a perf matrix.
  6. Consider impact of data locality
     

## Using the profiler

python offline_profiler.py <list of profiler options>  <training script>  <additional training script specific args>
  
  1.  If an argument is repeated in both profiler options and train script options, profiler option is chosen
  2.  Profiler_options
      * --docker-img  : docker container image name (full path so that it can be downloaded if not present locally 
      * --container-mnt: mountpoint in the container where dataset will be mounted. Default : /datadrive
      * --num-gpus    : number of GPUs used in training
      * --nnodes      : number of nodes used in training
      * --master_addr : IP of master node (same as torch.distr.launch)
      * --master_port : Free port on master (same as torch.distr.launch)
      * --cpu         : Max # CPUs to use. If not specified, uses all available CPUs on the server
      * --memory      : Max memory (GB) to be used in profiling. If not specified, uses max DRAM on system
      * -b            : Per-GPU batch size
      * --max-iterations: Max iterations per profiling run
      
  3. The following options are expected to be supported by the job script. 
      * --batch, -b   : Batch size per GPU
      * --workers, -j : Number of CPU data workers
      * --max_iterations: Return training after these # of iterations
      * --data        : Path to dataset

List of all other supported options can be found using the following command
```
python offline_profiler.py -h
```

### Example

```
 -  cd store-aware-packing/src
 -  cd profiler; ./prereq.sh; cd ..
 -  python profiler/offline_profiler.py --job-name job-res18 --cpu 24 --num-gpus 4 -b 512 --docker-img=nvidia/dali:py36_cu10.run ../models/image_classification/pytorch-imagenet.py --dali --amp --dali_cpu --max-iterations 50 --workers 3 -b 512 --data '/datadrive/mnt4/jaya/datasets/imagenet/' | tee res18out.log
```
