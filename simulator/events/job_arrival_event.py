from events.event_type import EventType
from events.cluster_event import ClusterEvent
import time

class JobArrivalEvent(ClusterEvent):

    def __init__(self, time, job):
        super().__init__(
            time, 
            int(EventType.JOB_ARRIVAL))
        self.job = job

    def handleEvent(self):
        super().handleEvent()
        ClusterEvent.runner.start_job(self.job)
        if ClusterEvent.runner.simulate and ClusterEvent.runner.trace is None:
     
            if ClusterEvent.runner.num_jobs_default > 0:
                if ClusterEvent.runner.num_jobs_so_far < ClusterEvent.runner.num_jobs_default:
                    if not ClusterEvent.runner.static:
                        ClusterEvent.runner.add_next_job()
                    else:
                        ClusterEvent.runner.add_next_job(arrival=0)

            else:
                ClusterEvent.runner.add_next_job()

    def __str__(self):
        return "event:{}:{:.2f}:{}:{}".format(self.time, time.time() -ClusterEvent.runner.real_start_time, EventType.get_name(self.event_type), self.job)
