from events.event_type import EventType
from events.cluster_event import ClusterEvent
import time

class JobEndEvent(ClusterEvent):

    def __init__(self, time, job):
        super().__init__(
            time, 
            int(EventType.JOB_END))
        self.job = job

    def handleEvent(self):
        super().handleEvent()
        runner = ClusterEvent.runner
        cluster = runner.cluster

        self.job.job_last_execution_time = self.time

        # deallocate gpus from jobs on cluster
        cluster.deallocate(self.job.gpus, self.job)

        # finish job and runner state update
        runner.finish_job(self.job)

    def __str__(self):
        return "event:{}:{:.2f}:{}:{}".format(self.time,time.time() -ClusterEvent.runner.real_start_time, EventType.get_name(self.event_type), self.job)
