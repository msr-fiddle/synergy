from events.event_type import EventType
from events.cluster_event import ClusterEvent
from events.schedule_event import ScheduleEvent
import time

class JobLeaseEndEvent(ClusterEvent):

    def __init__(self, time, job, is_finished=False):
        super().__init__(
            time, 
            int(EventType.JOB_LEASE_END))
        self.job = job
        self.is_finished = is_finished

    def handleEvent(self):
        super().handleEvent()
        runner = ClusterEvent.runner
        cluster = runner.cluster

        self.job.job_last_execution_time = self.time
        if not runner.simulate:
            self.logger.info("Dealloc job {} : GPUs : {}, CPUs : {}, Mem : {}, time={}, iters={}, cur_time={}".format(str(self.job), self.job.gpus, self.job.cpus, self.job.mem, self.job.attained_service_time, self.job.job_executed_iteration, runner.get_time()))
#        else:
#            self.logger.info("Lease end for job {} : GPUs : {}, CPUs : {}, Mem : {}, time={}, iters={}, cur_time={}".format(str(self.job), self.job.gpus, self.job.cpus, self.job.mem, self.job.attained_service_time, self.job.job_executed_iteration, runner.get_time()))

        # deallocate gpus from job on cluster
        if not  runner.scheduler.opt:
            cluster.deallocate(self.job.gpus, self.job)

     
        # job finish check and runner state update
        if self.job.is_finished() or self.is_finished:
            runner.finish_job(self.job)

    def __str__(self):
        return "event:{:.2f}:{:.2f}:{}:{}".format(self.time,time.time() -ClusterEvent.runner.real_start_time, EventType.get_name(self.event_type), self.job)
