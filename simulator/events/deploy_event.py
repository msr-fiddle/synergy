from events.event_type import EventType
from events.cluster_event import ClusterEvent
from events.allocation_event import AllocationEvent
import copy

class DeployEvent(ClusterEvent):

    def __init__(self, time, scheduler):
        super().__init__(
            time, 
            int(EventType.DEPLOY))
        self.scheduler = scheduler

    def handleEvent(self):
        super().handleEvent()
        runner = ClusterEvent.runner
        all_jobs_map = {}
        for job in runner.runnable_jobs:
            all_jobs_map[job.job_id] = job

        self.logger.info(all_jobs_map)
        job_ids_this_round = self.scheduler.running_job_ids
        #jobs = self.scheduler.running_jobs
        jobs = self.scheduler.running_jobs
        self.logger.info("Sched status prior to deploy : {}".format(runner.done_sched_next_round._value))

        if len(runner.runnable_jobs) == 0:
            runner.terminate = True
            return        

        #self.logger.info("Deploy Job status : ")
        #for job in jobs:
        #    self.logger.info("\t{}".format(str(job)))


        if not runner.simulate :
            # Wait for async allocaton to finish
            while runner.done_sched_next_round._value != 1:
               continue

            # Wait for current round jobs to report
            new_jobs = self.scheduler.running_jobs
            #prev_job_ids = [job.job_id for job in self.scheduler.prev_round_jobs]
            #prev_jobs = [job for job in new_jobs if job.job_id in prev_job_ids]
            self.logger.info("Prev round jobs = {}".format(self.scheduler.prev_round_jobs))
            for old_job in self.scheduler.prev_round_jobs:
                if old_job.job_id not in all_jobs_map:
                    self.logger.info("Prev Job {} NOT in map :  current status : time={}, iters={}/{}".format(str(old_job), old_job.attained_service_time, old_job.job_executed_iteration, old_job.job_total_iteration))
                    self.scheduler.prev_round_jobs.remove(old_job)
                else:
                    job = all_jobs_map[old_job.job_id]
                    self.logger.info("Prev Job {} current status : time={}, iters={}/{}".format(str(job), job.attained_service_time, job.job_executed_iteration, job.job_total_iteration))

            if len(self.scheduler.prev_round_jobs) > 0:
                self.logger.info("Waitinf for reporting {} : {}".format(all_jobs_map, self.scheduler.prev_round_jobs))
                while not self.have_reported(self.scheduler.prev_round_jobs, all_jobs_map):
                    continue
        
            self.logger.info("Round End Report before : {}".format(runner.round_end_report))
            
            runner.ready_to_deploy_next_round.inc()


            if len(self.scheduler.prev_round_jobs) > 0:
                while not self.have_reported(self.scheduler.prev_round_jobs, all_jobs_map, finish=True):
                    continue

            self.logger.info("Round End Report after : {}".format(runner.round_end_report))

            # TODO: Wait until all jobs have received lease update/terminate 
            # notifications and marked round end
 
            # Now, for those jobs whose lease ended, let the sched handle 
            # leaseEndEvent 
            

            # Identify jobs whose allocation hasnt changed
            # New jobs with allocations specified

            # Check jobs that have finished since allocation
            self.logger.info("Jobs completed this round = {}".format(runner.job_ids_finished_this_round))
            runner.remove_finished_jobs()

            self.logger.info("Job IDs to run = {}".format(runner.job_ids_to_run))
            jobs_to_run = [job for job in new_jobs if job.job_id in runner.job_ids_to_run]
            #jobs_to_run = self.scheduler.get_new_jobs(old_jobs, new_jobs)
            runner.deploy_ongoing.inc()
            self.logger.info("Jobs to deploy : ")
            for job in jobs_to_run:
                self.logger.info(str(job))

            self.logger.info("Job Lease Status : {}".format(runner.job_lease_status))
            self.scheduler.deploy_jobs_round(jobs_to_run)

            next_half_round_time = runner.get_time() + self.scheduler.round_duration*0.8
            next_round_time = runner.get_time() + self.scheduler.round_duration
            
            runner.add_event(AllocationEvent(next_half_round_time, self.scheduler))
            runner.add_event(DeployEvent(next_round_time, self.scheduler))



            
            
            runner.done_sched_next_round.dec()
            runner.ready_to_deploy_next_round.dec()
            runner.deploy_ongoing.dec()

            self.logger.info("returning from deploy... Sched status after deploy : {}".format(runner.done_sched_next_round._value))

    # Compare ClusterEvent.runner.round_end_report and job's GPU counts
    def have_reported(self, jobs, all_jobs_map, finish=False):
        with ClusterEvent.runner.scheduler_lock:
            report = copy.deepcopy(ClusterEvent.runner.round_end_report)

        for job in jobs:
            if job.job_id not in report:
               if not finish:
                 #if job.job_id in all_jobs_map and not all_jobs_map[job.job_id].is_finished():
                 if job.job_id in all_jobs_map and job.job_id not in ClusterEvent.runner.job_ids_finished_this_round:
                      del report
                      return False
                 else:
                      continue
               else:
                   continue
            else:
               if not finish and report[job.job_id] != len(job.gpus):
                  #self.logger.info("Num iterators={}, reported={}".format(len(job.gpus), report[job.job_id]))
                  del report
                  return False
               if finish and report[job.job_id] >= 0:
                  del report
                  return False
        del report
        return True
