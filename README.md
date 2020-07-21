## Synergy : Workload- and Data-Aware Scheduling for Multi-Tenant GPU Clusters
GPU cluster schedulers for DNN jobs decide how to allocate diverse resources to many users while implementing
complex cluster-wide scheduling policies to optimize for objectives such as average job completion
times (JCT), makespan, or user-level fairness. All prior work on DNN scheduling assume GPU to be the dominant
resource in the scheduling task; i.e., a user requests a fixed number of GPUs for her DNN job, and
when the requested number of GPUs are all available at once (gang scheduling), the job is scheduled
to run. However, the GPU cluster schedulers as of today ignore two important characteristics of DNN
jobs that results in inefficient job placements and reresource allocation.
  * **Resource sensitivity**. GPU cluster schedulers do not consider the sensitivity of a DNN job to
the CPU, DRAM, and storage resources (data stall) allocated to the job. These resources are
simply fair-shared proportional to the number of GPUs allocated to the job.
  * **Data sensitivity**. Prior work on GPU cluster scheduling ignore the impact of dataset locality;
they assume that the dataset required by the job is present locally before the training begins
and are small enough to fit in DRAM.

This work introduces a new DNN scheduler, **Synergy**, which shows that DNN jobs exhibit varied levels of sensitivity to the amount of CPU,
DRAM, and storage allocated to the job; the scheduler must be cognizant of the job resource requirements 
(beyond just the GPU), and allocate storage resources proportional to the workload requirements
rather than a fair-share. Furthermore, input datasets are typically stored on remote stores like Azure
blobs which has to be imported locally when a job is scheduled to run on a server. We show that
schedulers must be mindful of data locality when performing job migrations either due to re-packing
to fit newly arriving jobs, or a better performing schedule is identified.


[[PPT]](https://drive.google.com/file/d/1bBibeGadySMbhZZRZOec2CEbiMsBwKaL/view?usp=sharing)
[[Experiment Sheet]](https://drive.google.com/file/d/15gPVzWU4ThVi1isqtDIGsw7RHNm_8AiE/view?usp=sharing)

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
