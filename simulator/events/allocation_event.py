from events.event_type import EventType
from events.cluster_event import ClusterEvent
import copy

class AllocationEvent(ClusterEvent):

    def __init__(self, time, scheduler):
        super().__init__(
            time, 
            int(EventType.ALLOCATION))
        self.scheduler = scheduler

    def handleEvent(self):
        super().handleEvent()

        runner = ClusterEvent.runner
        jobs = copy.deepcopy(runner.runnable_jobs)

        start_time = runner.get_time()
        self.logger.info("Starting allocation at :{}s".format(runner.get_time())) 
        # Clear prior allocation
        for job in jobs:
            job.clear_alloc_status(runner.simulate)
            self.logger.info("Job {} status at allocation : time={}, iters={}/{}".format(str(job), job.attained_service_time, job.job_executed_iteration,job.job_total_iteration))

        gpus = runner.cluster.get_free_gpus()

        next_round_time = runner.get_time() + self.scheduler.round_duration
        
        self.logger.info("GPU status : Free GPUs = {} ".format(len(gpus)))
        #self.logger.info("Runnable jobs : ")
        #for job in jobs:
        #    self.logger.info("\t{}".format(str(job)))
        
        cluster_copy = copy.deepcopy(runner.cluster_cleanstate)
        self.scheduler.schedule(jobs, gpus, cluster_copy=cluster_copy)

        self.logger.info("Allocated jobs : ")
        for job in self.scheduler.running_jobs:
            gpu_ids = [gpu.gpu_id for gpu in job.gpus]
            self.logger.info("\t{}:g{}:c{}:m{}".format(str(job),gpu_ids, job.cpus, job.mem))

                
        old_jobs = self.scheduler.prev_round_jobs
        new_jobs = self.scheduler.running_jobs
        with runner.scheduler_lock:
            runner.job_ids_to_run = self.scheduler.lease_update(old_jobs, new_jobs)
        
        runner.done_sched_next_round.inc()
        self.logger.info("Sched status : {} at {}s. Took {}s".format(runner.done_sched_next_round._value, runner.get_time(), runner.get_time() - start_time))

