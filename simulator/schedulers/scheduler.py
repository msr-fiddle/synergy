from enum import Enum
import logging
from resources.cluster import AllocationStrategy
import sys
#from deployment.synergy_iterator import SynergyIterator 
#from deployment.runtime.rpc.scheduler_client import JobDescription 
#from deployment.helper import get_self_ip, list_as_string 
import os
import copy
import time

SYNERGY_CPU_THIS_SERVER = 'SYNERGY_CPU_THIS_SERVER'
SYNERGY_GPU_THIS_SERVER = 'SYNERGY_GPU_THIS_SERVER'
SYNERGY_MEM_THIS_SERVER = 'SYNERGY_MEM_THIS_SERVER'
SYNERGY_ARCH= 'SYNERGY_ARCH'
SYNERGY_TOTAL_ITERS = 'SYNERGY_TOTAL_ITERS'
SYNERGY_BATCH_SIZE = 'SYNERGY_BATCH_SIZE'
SYNERGY_GPUS_ALLOCATED = 'SYNERGY_GPUS_ALLOCATED'
SYNERGY_CPUS_ALLOCATED = 'SYNERGY_CPUS_ALLOCATED'
SYNERGY_LOG_DIR = 'SYNERGY_LOG_DIR'
SYNERGY_CHK_DIR = 'SYNERGY_CHK_DIR'
SYNERGY_ELAPSED_ITERS = 'SYNERGY_ELAPSED_ITERS'
SYNERGY_DISTRIBUTED = 'SYNERGY_DISTRIBUTED'
SYNERGY_MASTER_IP = 'SYNERGY_MASTER_IP'
SYNERGY_MASTER_PORT = 'SYNERGY_MASTER_PORT'
SYNERGY_NNODES = 'SYNERGY_NNODES'
SYNERGY_RANK = 'SYNERGY_RANK'
SYNERGY_START_RANK = 'SYNERGY_START_RANK'
SYNERGY_WORLD_SIZE = 'SYNERGY_WORLD_SIZE'

class SynergyMode(Enum):
    LOCAL = 1
    GLOBAL = 2
    STRICT = 3
    ADJUST = 4

class Scheduler:
    
    runner = None

    def __init__(
        self, 
        preemption=False,
        round_duration=300,
        placement=True,
        tenant_group=0,
        fair=True,
        tune=False,
        opt=False,
        simulate=True):
        self.preemption = preemption
        self.round_duration = round_duration
        self.tenant_group = tenant_group
        self.placement = placement
        self.running_jobs = []
        self.running_job_ids = []
        self.prev_round_jobs = []
        self.fair = fair
        self.tune = tune
        self.opt = opt
        self.simulate = simulate
        self.logger = logging.getLogger(__name__)
        if self.placement:
            #self.alloc_strategy = AllocationStrategy.PLACEMENT_SENSITIVE
            self.alloc_strategy = AllocationStrategy.SYNERGY_PLACEMENT
        else:
            self.alloc_strategy = AllocationStrategy.SYNERGY_RANDOM
            #self.alloc_strategy = AllocationStrategy.DEFAULT_ORDER
        self.logger.info("[SCHEDULER] : placement={}, fair={}, tune={}, opt={}, alloc_strategy={}".format(self.placement,self.fair,self.tune, self.opt, self.alloc_strategy))
        

    
    def schedule(self, jobs, gpus):
        pass

    def get_current_job_by_id(self, job_id, option=None):

        if self.runner.done_sched_next_round._value:
            if option == 0 or option is None:
                for job in self.prev_round_jobs:
                    if job.job_id == job_id:
                        return (job, 0)
          
        if option == 1 or option is None:
            for job in self.running_jobs:
                if job.job_id == job_id:
                    return (job, 1)

        return (None, -1)



    # Finds diff in allocation between prev_round and current round
    # and populates the job_lease_status map for jobs whose allocation hasnt changed
    # and returns a list of jobs to be run afresh this round
    def lease_update(self, old_jobs, new_jobs):
        #old_jobs = self.prev_round_jobs
        #new_jobs = self.running_jobs
        job_ids_to_run = list()
        old_job_map = dict()
        for job in old_jobs:
            old_job_map[job.job_id] = job


        for job in new_jobs:
            if job.job_id not in old_job_map:
                job_ids_to_run.append(job.job_id)
            else:
                old_job = old_job_map[job.job_id]
                new_job = job

                # Find diff in allocation
             #   old_res_map = old_job.res_map
             #   new_res_map = new_job.res_map
   
             #   old_alloc_map = dict()
             #   new_alloc_map = dict()

             #   for serv in old_res_map.keys():
             #       old_alloc_map[serv.server_id] = old_res_map[serv]['gpu']
             #   for serv in new_res_map.keys():
             #       new_alloc_map[serv.server_id] = new_res_map[serv]['gpu']

             #   same_serv_gpus = False
             #   if list(old_alloc_map.keys()).sort() == list(new_alloc_map.keys()).sort():
             #      same_serv_gpus = True
             #      for serv in old_alloc_map.keys():
             #          if old_alloc_map[serv] != new_alloc_map[serv]:
             #              same_serv_gpus = False


                old_gpu_ids = [gpu.gpu_id for gpu in old_job.gpus] 
                new_gpu_ids = [gpu.gpu_id for gpu in new_job.gpus]
                old_gpu_ids.sort()
                new_gpu_ids.sort()
                self.logger.info("Job:{}, old={}, new={}".format(str(job), old_gpu_ids, new_gpu_ids))

                if old_gpu_ids == new_gpu_ids:
                #if old_gpu_ids == new_gpu_ids or same_serv_gpus:
                    if old_job.cpus == new_job.cpus and old_job.mem == new_job.mem:
                        # Extend lease
                        self.runner.job_lease_status[job.job_id] = len(job.gpus)
                        # Copy CPU IDs
                        new_job.cpu_ids = old_job.cpu_ids

                        for server in new_job.res_map:
                            for server_id in new_job.cpu_ids:
                                if server_id == server.server_id:
                                    cpu_ids_to_remove = new_job.cpu_ids[server_id]
                                    server.remove_cpus_available(cpu_ids_to_remove)
                                    break
                        self.logger.info("Job:{}, old CPUs={}, new CPUs={}".format(str(job), old_job.cpu_ids, new_job.cpu_ids))
                        for server in new_job.res_map:
                          self.logger.info("Job:{}, Server:{}, new server CPUs={}".format(str(job), server.server_id, server.print_cpus_available()))
                        for server in old_job.res_map:
                            self.logger.info("Job:{}, Server:{}, old server CPUs={}".format(str(job),server.server_id, server.print_cpus_available()))
                        continue


                self.logger.info("Job:{}, old CPUs={}, new CPUs={}".format(str(job), old_job.cpu_ids, new_job.cpu_ids))
                for server in new_job.res_map:
                    self.logger.info("Job:{}, Server:{}, new server CPUs={}".format(str(job), server.server_id, server.print_cpus_available()))
                for server in old_job.res_map:
                    self.logger.info("Job:{}, Server:{}, old server CPUs={}".format(str(job),server.server_id, server.print_cpus_available()))

                if job.job_id in self.runner.job_lease_status:
                    del self.runner.job_lease_status[job.job_id]

                job_ids_to_run.append(job.job_id)
        self.logger.info("Job Ids to run = {}".format(job_ids_to_run))
        return job_ids_to_run
                
        


    # Deploy new jobs 
    def deploy_jobs_round(self, jobs_to_run):

        if len(jobs_to_run) == 0:
            self.logger.info("No new jobs to deploy. Lease extended for {} jobs".format(len(self.prev_round_jobs)))
            return

        self.jobs_to_run = copy.deepcopy(jobs_to_run)
        #self.jobs_to_run = copy.deepcopy(self.running_jobs)
        if len(self.jobs_to_run) > 0:
            self.logger.info("Jobs to Deploy this round : ")
        for job in self.jobs_to_run:
             #job_descr  = self.dummy_job(job)
             #self.runner.get_job_by_id(job.job_id).copy_with_alloc_status(job, simulate=False)

             #time.sleep(5)
             distributed = False
             master_ip = 0
             master_port = 0
             nnodes = 0
             gpu_ids = {}
             if len(job.gpus) > 1:
                 distributed = True
                 server_list = list(job.res_map.keys())
                 server_list.sort(key=lambda s: s.server_id)
                 master_ip = server_list[0].ip
                 master_port = server_list[0].port + 1000 + job.job_id
                 nnodes = len(server_list)
                 self.logger.info("Distributed : ip={}, port={}, nnodes={}".format(master_ip,master_port,nnodes))
             start_rank = 0
             world_size = len(job.gpus)
             for i, server in enumerate(job.res_map.keys()):
                 #self.logger.info("Server CPU list = {}".format(server.print_cpus_available()))
                 if distributed:
                     job_descr, ids = self.create_job_descr(job, server, i, nnodes=nnodes, ip=master_ip, port = master_port, start_rank=start_rank, world_size=world_size)
                     start_rank += job.res_map[server]['gpu']
                 else:
                     job_descr, ids = self.create_job_descr(job, server, i)
                 gpu_ids[server.server_id] = ids
                 server.rpc_client.run_job(job_descr, 0)
             self.logger.info("Deploying job {}\n\tGPUs={}\n\tResMap={}\n\tCPU_IDs={}\n\tGPU_IDs={} at {} s".format(\
                 str(job),\
                 job.gpus,\
                 job.res_map,\
                 job.cpu_ids,\
                 gpu_ids,\
                 self.runner.get_time()))
             self.logger.info("Returned from launch..")

    # Job description per server
    def create_job_descr(self, job, server, rank, nnodes=1, ip=0, port=0, start_rank=0, world_size=1):
        serv_job_res = job.res_map[server]
        new_env = copy.deepcopy(os.environ)
        job_id = job.job_id

        job_alloc_version, status = self.get_current_job_by_id(job_id, option=1)

        # GPU objects are numbered consecutively across servers. So index appropriately
        # into each server's GPU devices for deployment
        gpu_ids_available = [gpu.gpu_id for gpu in server.gpus]
        gpu_ids_allocated = [gpu.gpu_id for gpu in job.gpus]
        gpu_ids_to_use = set(gpu_ids_allocated).intersection(gpu_ids_available)
        
        gpu_ids = [(gpu_id %server.num_gpus) + server.start_gpu_deploy for gpu_id in gpu_ids_to_use]
        #gpu_ids = [(gpu.gpu_id %server.num_gpus) + server.start_gpu_deploy for gpu in job.gpus]
        num_cpu = int(serv_job_res['cpu'])
        cpu_ids = server.get_cpus(num_cpu)
        #self.logger.info("GPU ids ={}, CPUs={} for server {}".format(gpu_ids, cpu_ids, server.server_id))
        for alloc_server in job_alloc_version.res_map:
            if alloc_server.server_id == server.server_id:
                alloc_cpu_ids = alloc_server.get_cpus(num_cpu)
                self.logger.info("New CPU ids={}, alloc cpu ids = {}".format(cpu_ids, alloc_cpu_ids))
                break

        job.cpu_ids[server.server_id] = cpu_ids
        job_alloc_version.cpu_ids[server.server_id] = cpu_ids

        self.logger.debug("Info for job {} and server {}".format(job.job_id, server.server_id))
        self.logger.debug(gpu_ids) 
        self.logger.debug("Num GPU={}".format(serv_job_res['gpu']))
        self.logger.debug("Num CPU={}".format(serv_job_res['cpu']))
        self.logger.debug("Num mem={}".format(serv_job_res['mem']))
        self.logger.debug("Num sspeed={}".format(serv_job_res['sspeed']))
        new_env[SYNERGY_CPU_THIS_SERVER] = str(serv_job_res['cpu'])
        new_env[SYNERGY_MEM_THIS_SERVER] = str(serv_job_res['mem'])
        new_env[SYNERGY_GPU_THIS_SERVER] = str(serv_job_res['gpu'])
        new_env[SYNERGY_ELAPSED_ITERS] = str(job.job_executed_iteration)
        new_env[SYNERGY_ARCH] = str(job.job_model.model_name)
        new_env[SYNERGY_TOTAL_ITERS] = str(job.job_total_iteration)
        new_env[SYNERGY_BATCH_SIZE] = str(job.job_model.batch_size)
        new_env[SYNERGY_GPUS_ALLOCATED] = ",".join(list_as_string(gpu_ids))
        new_env[SYNERGY_CPUS_ALLOCATED] = ",".join(list_as_string(cpu_ids))
        new_env[SYNERGY_LOG_DIR] = "./logs/"

        cmd_script = "./docker_script_singlegpu.sh ./sched-res/  "

        if len(job.gpus) > 1:
            new_env[SYNERGY_DISTRIBUTED] = "1"
            new_env[SYNERGY_NNODES] = str(nnodes)
            new_env[SYNERGY_MASTER_IP] = str(ip)
            new_env[SYNERGY_MASTER_PORT] = str(port)
            new_env[SYNERGY_RANK] = str(rank)
            new_env[SYNERGY_START_RANK] = str(start_rank)
            new_env[SYNERGY_WORLD_SIZE] = str(world_size)
            cmd_script = "./docker_script_multigpu.sh ./sched-res/  "

     
        self.logger.info("Job {}, iters={}/{}".format(job.job_id, job.job_executed_iteration, job.job_total_iteration))        
        cmd = cmd_script + \
              str(get_self_ip()) + \
              " " + \
              str(self.runner.sched_port) + \
              " " + \
              str(job_id)

        #TODO : Test
        #cmd = "./test_multigpu.sh ./out/ 10.185.12.207 14000 25 "

        job_description = JobDescription( \
                            job_id, \
                            cmd, \
                            work_dir="./", \
                            env=new_env)
        return job_description, gpu_ids

   
    def dummy_job(self, job):
        new_env = copy.deepcopy(os.environ)
        job_id = 111
        gpu_ids = [gpu.gpu_id for gpu in job.gpus] 
        cmd = "./dummy_docker_script.sh ./sched-res/  " + str(get_self_ip()) + " " + str(self.runner.sched_port) + " " + str(job_id)
        job_description = JobDescription(job_id, cmd, work_dir="./",env=new_env)  
        return job_description
