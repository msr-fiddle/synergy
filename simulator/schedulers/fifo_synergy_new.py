import logging
import numpy as np
from schedulers.scheduler import Scheduler
from events.job_lease_end_event import JobLeaseEndEvent
from opt_algo.synergy_opt import Solver
import copy
import time

class FIFO(Scheduler):

    def __init__(
        self,
        preemption=True,
        round_duration=300,
        placement=True, 
        tenant_group=0,
        fair=True,
        tune=False,
        opt=False,
        simulate=True):
        self.logger = logging.getLogger(__name__)
        super().__init__(preemption, round_duration, placement, tenant_group, fair, tune, opt, simulate)

    def schedule(self, jobs, gpus, cluster_copy=None):
        free_gpus = gpus
        runner = Scheduler.runner
        if runner.simulate:
            cluster = runner.cluster
        else:
            cluster = cluster_copy
            free_gpus = cluster.get_free_gpus()

        runner_time = runner.get_time()
        self.prev_round_jobs = copy.deepcopy(self.running_jobs)
        self.running_jobs = []
        self.running_job_ids = []
        skipped_jobs = []
        
        jobs.sort(key=lambda job: job.job_arrival_time)

        cluster.server_job_schedule = [list() for server in cluster.servers]

        for job in jobs:
            job.synergy_speedup = job.synergy_speedup_orig

        jobs_this_round = []
        num_free_gpus = len(free_gpus)
        for job in jobs:
            job_gpu_deficit = job.get_gpu_deficit()
            if job_gpu_deficit > 0 and num_free_gpus >= job_gpu_deficit:
                jobs_this_round.append(job)
                num_free_gpus -= job_gpu_deficit
            if job.get_time_since_last_execution(runner_time) > 1000 * 3600:
                job.job_priority = 1
        start_time = time.time()
        if self.opt and self.simulate:
            if len(jobs_this_round) <= 0:
                return 
 
            for job in jobs_this_round:
                self.running_jobs.append(job)
            num_servers = len(cluster.servers)
            per_server_size = cluster.per_server_size
            solver = Solver(
                num_servers = num_servers,
                gpu_per_server=per_server_size[0],
                cpu_per_server=per_server_size[1],
                mem_per_server=per_server_size[2],
                debug = True,
                round_dur=self.round_duration)
            solver.SolveFirstLP(self.running_jobs, ilp=True, solver='GLPK_MI')
            solver.SolveSecondLP(ilp=False, solver='GLPK_MI')
            self.logger.info("Time to alloc = {}s".format(time.time() - start_time))
   
            #if solver.is_fractional:
            #    solver.RoundRobin()             
            for job in self.running_jobs:
                solver.allocate_round(job)         

                job_lease_end_time = runner_time + self.round_duration
                runner.add_event(JobLeaseEndEvent(job_lease_end_time, job))
            return
      


        for job in jobs:
            self.logger.debug(job)
            assert(len(free_gpus) == len(cluster.get_free_gpus(free_gpus)))
            job_gpu_deficit = job.get_gpu_deficit()

            if job_gpu_deficit > 0 and len(free_gpus) >= job_gpu_deficit:
                success, free_gpus = cluster.allocate(
                    free_gpus, job_gpu_deficit, job, self.round_duration, 
                    self.alloc_strategy, fair=self.fair, tune=self.tune)
                if not success:
                    skipped_jobs.append(str(job))
                else:
                    self.running_jobs.append(job)
                    self.running_job_ids.append(job.job_id)
                    if self.simulate:
                        job_lease_end_time = runner_time + self.round_duration
                        runner.add_event(JobLeaseEndEvent(job_lease_end_time, job))
                    else:
                        runner.sched_job_threshold += 1

            if job.get_time_since_last_execution(runner_time) > 1000 * 3600:
                job.job_priority = 1

                # TODO : If leases > a round and a job has had its lease extended, 
                # then gpu_deficit for the job =0 but its running. Account for this later.

                #self.logger.info("[{}] Model={}, GPU={}, CPU demand={}, deficit = {}".format(job.job_id, job.job_class_id, job.job_gpu_demand, job.job_cpu_demand, job.get_cpu_deficit))
                # get cpu, mem requirements of running jobs
                    
