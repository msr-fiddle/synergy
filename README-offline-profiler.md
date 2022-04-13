## Looking Beyond GPUs for DNN Scheduling for Multi-Tenant GPU Clusters

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

### Run instructions

Synergy docker 
```
 - git clone https://github.com/jayashreemohan29/Synergy-CoorDL.git
 - git checkout iterator_chk
 - cd docker
 - CREATE_RUNNER="YES" ./build.sh
 This will create a docker container tagged nvidia/dali:py36_cu10.run

```
 -  cd synergy-private/src
 -  cd profiler; ./prereq.sh; cd ..
 -  If you have issues using the docker container, please ask us and we shall give a docker tar (~8GB, hence not uploaded here). You can load the docker container with dependencies installed : docker load -i dali_docker.tar
 -  python profiler/offline_profiler.py --job-name job-res18 --cpu 24 --num-gpus 4 -b 512 doiiii--docker-img=nvidia/dali:py36_cu10.run ../models/image_classification/pytorch-imagenet.py --dali --amp --dali_cpu --max-iterations 50 --workers 3 -b 512 --data '/datadrive/mnt4/jaya/datasets/imagenet/' | tee res18out.log
 - or execute ./run-cnns.sh to profile all image classification models. Please update dataset path appropriately in the script
```
