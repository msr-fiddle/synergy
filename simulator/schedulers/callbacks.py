import sys
import os
import logging
import math
import time
import threading

from events.job_lease_end_event import JobLeaseEndEvent
from deployment.runtime.rpc import scheduler_client


class SchedulerCallbacks():
    def __init__(self, launcher):
        self.launcher = launcher
        self.logger = logging.getLogger(__name__)
        if launcher.cluster is None:
            raise ValueError("Custer struct unavailable to populate")

        self.cluster = self.launcher.cluster
        self.lock = self.launcher.scheduler_lock
        
        """
         A map of job ID vs round number that indicates if the updates for the given round
         are performed by any iterator of a job. Updates include job time/iter 
         updates to the Job object. Used to synchronize across iterators of a job
        """
         
        self.job_metrics_update_map = dict()

        # Map of job ID vs iters for the current round.
        # Used to synchrnize across iterators of a multi-GPU job
        self.job_iters = dict()

        # Map of job ID to lease renewal status for the round
        self.job_lease_status = self.launcher.job_lease_status

        self.round_end_report = self.launcher.round_end_report

    """
    Called by a worker to register itself with the scheduler

    Registers a worker (ip, port) with `num_gpus` GPUs
    An RPC client is set up using the given (IP, Port) to enable
    communication between the scheduler and the worker to dispatch/kill jobs.
    """
    def register_worker_callback(self, num_gpus, ip_addr, port):
        rpc_client = scheduler_client.SchedulerRpcClient(ip_addr, port)
        server_handle = self.cluster.connection_to_server(ip_addr, port)
        try:
            server_handle.rpc_client = rpc_client
            if server_handle in self.cluster.machine_to_rpc_map:
                raise ValueError("Worker {} already registered with scheduler".format(server_handle.server_id))
            self.cluster.machine_to_rpc_map[server_handle] = rpc_client
            self.logger.info("Num workers so far = {}".format(len(self.cluster.machine_to_rpc_map)))
        except Exception as e:
            self.logger.error("Requested worker does not exist : {}".format(e))
            
        return self.launcher.round_duration, server_handle.server_id

    def register_job_callback(self, job_id):
        self.logger.info("Registered Job {} round dur = {} at {:.2f}s : {:.2f}s".format(job_id, self.launcher.round_duration, self.launcher.get_time() ,time.time()-self.launcher.real_start_time))
        if job_id in self.job_metrics_update_map:
            del self.job_metrics_update_map[job_id]
        if job_id in self.job_iters:
            del self.job_iters[job_id]
        return (1, self.launcher.round_duration)

    """
    Returns the number of iters the job can run for in the current round 
    Must be invoked by the iterator at 50-75% of the round duration, to ensure all
    iterators of the job run for a fixed number of iters per round
    (We estimate this every round because the profiled values are only for steady state.
    So estimate dynamically anyway)
    """
    def update_iters_callback(self, job_id, num_steps, execution_time_s, extend, round_num):

        with self.launcher.scheduler_lock:
            # If any previous iterator populated this map, then return the calculated value 
            if job_id in self.job_iters and \
               job_id in self.job_metrics_update_map and \
               self.job_metrics_update_map[job_id] == round_num:
                   return self.job_iters[job_id]

            time_per_iter = execution_time_s / num_steps
            steps_per_round = math.floor(self.launcher.round_duration / time_per_iter)
            self.logger.info("UpdateItersCallback: Job {} : time_per_iter={:.2f}, steps_per_round={}, round={} ".format(job_id,time_per_iter,steps_per_round, round_num))


            job_handle = self.launcher.get_job_by_id(job_id)
            job_handle.last_round_progress = job_handle.job_executed_iteration
            job_handle.last_round_attained_time = job_handle.attained_service_time
            job_handle.job_executed_iteration = job_handle.last_round_progress + steps_per_round
            job_handle.attained_service_time = job_handle.last_round_attained_time + self.launcher.round_duration

            self.job_metrics_update_map[job_id] = round_num
            self.job_iters[job_id] = steps_per_round

            # TODO :  Intiate next round scheduling
       
        return steps_per_round

    # returns lease termination status - True if terminated, False if extended for next round
    def  round_end_callback(self, job_id, num_steps, execution_time_s, done):
        with self.launcher.scheduler_lock:
            job_handle = self.launcher.get_job_by_id(job_id)
            self.logger.info("Job alloc status {} : GPUs : {}, CPUs : {}, Mem : {}, time={:.2f}, iters={}".format(str(job_handle), job_handle.gpus, job_handle.cpus, job_handle.mem, job_handle.attained_service_time, job_handle.job_executed_iteration))
            if job_id in self.launcher.round_end_report:
                self.launcher.round_end_report[job_id] += 1
            else:
                self.launcher.round_end_report[job_id] = 1

            #self.logger.info("Round End Report : ")
            self.logger.info("Round End Report : {}".format(self.launcher.round_end_report))

        self.logger.info("RoundEndCallback : Job {} : num_steps:{}, exec_time:{:.2f}, progress={}/{} at {:.2f}s:{:.2f}s".format(str(job_handle), num_steps, execution_time_s, job_handle.job_executed_iteration, job_handle.job_total_iteration, self.launcher.get_time() ,time.time()-self.launcher.real_start_time))
        self.logger.info("Done Sched : {}".format(self.launcher.done_sched_next_round._value))
        while not self.launcher.done_sched_next_round._value or not self.launcher.ready_to_deploy_next_round._value:
            continue
        self.logger.info("Done Sched : {}, ready_to_deploy={}".format(self.launcher.done_sched_next_round._value, self.launcher.ready_to_deploy_next_round._value))
        # TODO : if job_handle.is_finished() or job_id not in self.job_lease_status:
        self.logger.info("Job lease status before = {} ".format(self.launcher.job_lease_status))

        while self.launcher.deploy_ongoing._value:
            continue

        with self.launcher.scheduler_lock:

            if job_handle.is_finished() or job_id not in self.launcher.job_lease_status:
                self.logger.info("Job lease status after = {} ".format(self.launcher.job_lease_status))
                return True

            else:
                self.launcher.round_end_report[job_id] -= 1
                if self.launcher.round_end_report[job_id] <= 0:
                    del self.launcher.round_end_report[job_id]

        with self.launcher.scheduler_lock:
            self.launcher.job_lease_status[job_id] -= 1
            if self.launcher.job_lease_status[job_id] <= 0:
                del self.launcher.job_lease_status[job_id]

        self.logger.info("Extended Lease : Job lease status after = {} at {}s".format(self.launcher.job_lease_status, self.launcher.get_time()))
        return False

    def lease_ended_callback(self, job_id, steps_this_round, execution_time_s, round_num):
        job_handle = self.launcher.get_job_by_id(job_id)
        launcher_time = self.launcher.get_time()


        with self.launcher.scheduler_lock:
            # If any previous iterator populated this map, then return the calculated value 
            if  job_id not in self.job_metrics_update_map or \
               self.job_metrics_update_map[job_id] != round_num:
                    
                # Add last round steps and time
                job_handle.last_round_progress = job_handle.job_executed_iteration
                job_handle.last_round_attained_time = job_handle.attained_service_time
                job_handle.job_executed_iteration = job_handle.last_round_progress + steps_this_round
                job_handle.attained_service_time = job_handle.last_round_attained_time + execution_time_s     

                self.job_metrics_update_map[job_id] = round_num 
       

            #if job_id in self.launcher.round_end_report and steps_this_round > 0:
            #    self.launcher.round_end_report[job_id] += 1
            #elif steps_this_round > 0:
            #    self.launcher.round_end_report[job_id] = 1



 
        self.logger.info("Job alloc status {} : GPUs : {}, CPUs : {}, Mem : {}, time={:.2f}, iters={}".format(str(job_handle), job_handle.gpus, job_handle.cpus, job_handle.mem, job_handle.attained_service_time, job_handle.job_executed_iteration))

        is_finished = job_handle.is_finished()
        # To dealloc resources, get the version f job handle from scheduler with allocation status intact
        job_handle, status = self.launcher.scheduler.get_current_job_by_id(job_id)
        self.logger.info("Job alloc status in the alloc version of handle {} : GPUs : {}, CPUs : {}, Mem : {}, time={:.2f}, iters={}, status={}".format(str(job_handle), job_handle.gpus, job_handle.cpus, job_handle.mem, job_handle.attained_service_time, job_handle.job_executed_iteration, status))


        if job_handle.ready_to_deallocate():
            lease_end = JobLeaseEndEvent(launcher_time, job_handle, is_finished=is_finished)
            lease_end.handleEvent()
            #self.launcher.event_queue.put(JobLeaseEndEvent(launcher_time, job_handle, is_finished=is_finished)) 
            self.logger.info("LeaseEndCallback: Job [{}:{}] : progress:{}/{}, time={:.2f} at {:.2f}s:{:.2f}s, round={}".format(job_id, job_handle.job_model.model_name, job_handle.job_executed_iteration, job_handle.job_total_iteration, job_handle.attained_service_time, self.launcher.get_time() ,time.time()-self.launcher.real_start_time, round_num))


      
        with self.launcher.scheduler_lock:

            if job_id in self.launcher.round_end_report:
                self.launcher.round_end_report[job_id] -= 1
                if self.launcher.round_end_report[job_id] <= 0:
                    del self.launcher.round_end_report[job_id]

        return        
