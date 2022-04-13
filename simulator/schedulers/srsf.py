import logging
from schedulers.scheduler import Scheduler
from events.job_lease_end_event import JobLeaseEndEvent

class SRSF(Scheduler):
    def __init__(
        self,
        preemption=True,
        round_duration=300,
        placement=True,
        tenant_group=0):
        self.logger = logging.getLogger(__name__)
        super().__init__(preemption, round_duration, placement, tenant_group)
    
    def schedule(self, jobs, gpus):
        free_gpus = gpus
        runner = Scheduler.runner
        cluster = runner.cluster
        runner_time = runner.get_time()
        self.running_jobs = []

        jobs.sort(
            key=lambda job:
            (-job.job_priority, job.remaining_service(), job.job_arrival_time))

        for job in jobs:
            self.logger.debug(job)
            assert(len(free_gpus) == len(cluster.get_free_gpus(free_gpus)))
            job_gpu_deficit = job.get_gpu_deficit()
            if job_gpu_deficit > 0 and len(free_gpus) >= job_gpu_deficit:
                free_gpus = cluster.allocate(
                    free_gpus, job_gpu_deficit, job, self.round_duration, self.alloc_strategy)
                job_lease_end_time = runner_time + self.round_duration
                runner.add_event(JobLeaseEndEvent(job_lease_end_time, job))
                self.running_jobs.append(job)
            
            if job.get_time_since_last_execution(runner_time) > 1000 * 3600:
                # self.logger.info("Priority Bump for Job: %s", str(job.job_id))
                job.job_priority = 1
