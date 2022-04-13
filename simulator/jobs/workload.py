import os
import pickle
import math
import logging
import numpy as np
import sys
import pandas as pd
#from helpers import parse_philly_jobs
from helpers import parse_philly_trace_full
from helpers.utils import poisson_next_arrival_time, get_total_iteration, get_total_iteration_exp, get_job_gpu_demand, get_gavel_like_iter
from metrics.stats import DataSeries
from queue import PriorityQueue
from jobs.job import Job
from jobs.model_zoo import ModelZoo
from jobs.model_zoo import ModelAssignment
from jobs.task import TaskName

class Workload:

    def __init__(
        self,
        cluster_job_log,
        jobs_per_hour=5,
        sum_attempts=True,
        exponential=False,
        philly_arrival=False,
        multigpu=False,
        debug_multi=False,
        prioritize=False,
        small_trace=False,
        series_id_filter=(4000, 5000),
        model_class_split=(30,60,10),
        per_server_size=None,
        num_jobs_default=5,
        trace=None):
        self.logger = logging.getLogger(__name__)
        self.trace = trace

        if trace is not None:
            self.workload_type = "replay"
        elif cluster_job_log is None:
            self.workload_type = "synthetic"
        else:
            self.workload_type = "philly"
        
        self.logger.info("Workload Mode: %s", self.workload_type)

        self.job_id = 0
        self.jobs_per_hour = jobs_per_hour
        self.prioritize = prioritize
        self.series_id_filter = series_id_filter

        # Absolute size (num) resource vector per server
        self.per_server_size = per_server_size

        # In order - image, language, speech
        self.model_class_split = model_class_split

        # Model assignment option
        #self.model_assignment_option = ModelAssignment.FAIR
        #self.model_assignment_option = ModelAssignment.RANDOM
        self.model_assignment_option = ModelAssignment.OVERALL
        #self.model_assignment_option = ModelAssignment.RUNNABLE

        # choice of model, class to assign to the job
        self.model_zoo = ModelZoo(image_percent = model_class_split[0], 
                                  lang_percent = model_class_split[1],
                                  speech_percent = model_class_split[2],
                                  assignment = self.model_assignment_option)

        self.num_classes = len(self.model_zoo)
        self.philly_arrival = philly_arrival

        if self.workload_type == "philly":
            self.jobs = self.get_philly_jobs(cluster_job_log, sum_attempts, exponential, multigpu, debug_multi, small_trace)
            self.total_jobs = len(self.jobs)
            self.logger.info("Total Jobs in Philly Trace: %s, Exponential=%s, multigpu=%s, debug_multi=%s", self.total_jobs, exponential, multigpu, debug_multi)
        elif self.workload_type == "replay":
            self.jobs =  self.populate_from_trace(self.trace) 
            self.total_jobs = len(self.jobs)
            self.logger.info("Total Jobs in replay Trace: %s", self.total_jobs)
        else:
            self.jobs = self.default_workload(jobs=num_jobs_default)
            self.total_jobs = len(self.jobs)
            self.logger.info("Total Jobs in synthetic Trace: %s", self.total_jobs)
            for job in self.jobs:
                self.logger.info(str(job))
        
        

        self.logger.info("Available classes = {}".format(self.num_classes))
        self.logger.info("Chosen split = {}% image, {}% lang, {}% speech. Mode={}".format(*self.model_class_split, str(self.model_assignment_option)))

        #sys.exit(1)
        #self.placement_scores_by_class, self.packing_scores_by_class =\
        #    self.initialize_job_classes(self.num_classes)

    def add_runnable_job(self, job_class_id):
        self.model_zoo.add_runnable_job(job_class_id)

    def remove_runnable_job(self, job_class_id):
        self.model_zoo.remove_runnable_job(job_class_id)

    def get_philly_jobs(self, cluster_job_log, sum_attempts, exponential=False, multigpu=False, debug_multi=False, small_trace=False):
        jobs = []
        if not sum_attempts:
            if small_trace and not multigpu:
                fname = "philly_jobs_no_sum_attempts_static_single.pickle"
            elif small_trace and multigpu:
                fname = "philly_jobs_no_sum_attempts_static_multi.pickle"
            elif exponential and debug_multi:
                fname = "philly_jobs_no_sum_attempts_exp_multidebug.pickle"
            elif not exponential and debug_multi:
                fname = "philly_jobs_no_sum_attempts_multidebug.pickle"
            elif exponential and multigpu:
                fname = "philly_jobs_no_sum_attempts_exp_multi.pickle"
            elif exponential and not multigpu:
                fname = "philly_jobs_no_sum_attempts_exp_single.pickle"
            elif not exponential and not multigpu:
                fname = "philly_jobs_no_sum_attempts_single.pickle"
            elif not exponential and multigpu:
                fname = "philly_jobs_no_sum_attempts_multi.pickle"
            else:
                fname = "philly_jobs_no_sum_attempts.pickle"

            if os.path.exists(fname):
                jobs = pickle.load(
                    open(fname, "rb"))
            else:
                #jobs = parse_philly_jobs.parse_jobs(
                jobs = parse_philly_trace_full.parse_jobs_full(
                    cluster_job_log, sum_attempts, exponential, multigpu, debug_multi, small_trace, logger=self.logger)
                pickle.dump(
                    jobs, open(fname, "wb"))
        else:
            if small_trace and not multigpu:
                fname = "philly_jobs_sum_attempts_static_single.pickle"
            elif small_trace and multigpu:
                fname = "philly_jobs_sum_attempts_static_multi.pickle"
            elif exponential and debug_multi:
                fname = "philly_jobs_sum_attempts_exp_multidebug.pickle"
            elif not exponential and debug_multi:
                fname = "philly_jobs_sum_attempts_multidebug.pickle"
            elif exponential and multigpu:
                fname = "philly_jobs_sum_attempts_exp_multi.pickle"
            elif exponential and not multigpu:
                fname = "philly_jobs_sum_attempts_exp_single.pickle"
            elif not exponential and not multigpu:
                fname = "philly_jobs_sum_attempts_single.pickle"
            elif not exponential and multigpu:
                fname = "philly_jobs_sum_attempts_multi.pickle"
            else:
                fname = "philly_jobs_sum_attempts.pickle"
 
            self.logger.info("Reading job list file %s", fname)
            if os.path.exists(fname):
                jobs = pickle.load(
                    open(fname, "rb"))
            else:
                #jobs = parse_philly_jobs.parse_jobs(
                jobs = parse_philly_trace_full.parse_jobs_full(
                    cluster_job_log, sum_attempts,exponential, multigpu, debug_multi, small_trace, logger=self.logger)
                pickle.dump(
                    jobs, open(fname, "wb"))
        
        return jobs

    
    def add_synergy_profile(self, job):
        # get the corresponding model handle
        model_id = job.job_class_id
        job_model = self.model_zoo.model(model_id, job.job_gpu_demand)
        job.job_model = job_model
        job.synergy_speedup = job.job_model.speedup
        job.tput = job_model.tput
        update_speed = False
        try:
            job.synergy_res_matrix = job_model.synergy_res_score
            job.synergy_storage_matrix = job_model.synergy_storage_score
            job.job_placement_penalty = job_model.placement_penalty
            
            # Ideal CPU, mem, storage allocations
            job.job_cpu_demand_orig = job.job_gpu_demand * job_model.cpu_per_gpu
            job.job_mem_demand_orig = job.job_gpu_demand * job_model.mem_per_gpu
            job.job_sspeed_demand_orig = job.job_gpu_demand * job_model.sspeed_per_gpu

            

            if job.job_cpu_demand_orig > self.per_server_size[1]:
               cpu_ratio = self.per_server_size[1]/job.job_cpu_demand_orig
               job.job_cpu_demand_orig = self.per_server_size[1]
               update_speed = True
               if job.job_gpu_demand > self.per_server_size[0]:
                   job.job_cpu_demand_orig = job.job_gpu_demand *(self.per_server_size[1]/self.per_server_size[0])
                   job.synergy_speedup = 1
                   update_speed = False


            if job.job_mem_demand_orig > self.per_server_size[2]:
               mem_ratio = self.per_server_size[2]/job.job_mem_demand_orig
               job.job_mem_demand_orig = self.per_server_size[2]
               update_speed = True
               if job.job_gpu_demand > self.per_server_size[0]:
                   job.job_mem_demand_orig = job.job_gpu_demand *(self.per_server_size[2]/self.per_server_size[0])
                   job.synergy_speedup = 1
                   update_speed = False


            if job.job_sspeed_demand_orig > self.per_server_size[3]:
               sspeed_ratio = self.per_server_size[3]/job.job_sspeed_demand_orig
               job.job_sspeed_demand_orig = self.per_server_size[3]
               update_speed = True
               if job.job_gpu_demand > self.per_server_size[0]:
                   job.job_sspeed_demand_orig = job.job_gpu_demand *(self.per_server_size[3]/self.per_server_size[0])
                   job.synergy_speedup = 1
                   update_speed = False

            job.job_cpu_demand = job.job_cpu_demand_orig
            job.job_mem_demand = job.job_mem_demand_orig
            job.job_sspeed_demand = job.job_sspeed_demand_orig
        except:
            raise ValueError("No appropriate model found")


        # For now update perf based on cpu only. Real profile matrix will do acurtely
        # Approx: Fit a straight line between fair-share and ideal 
        if update_speed:
            speedup_diff = job.synergy_speedup - 1
            ideal_cpu = job.job_gpu_demand * job_model.cpu_per_gpu
            fair_cpu = job.job_gpu_demand *(self.per_server_size[1]/self.per_server_size[0])
            cpu_diff = ideal_cpu - fair_cpu
            if cpu_diff == 0:
                self.logger.info("{}: fair={}, ideal={}".format(str(job), fair_cpu, ideal_cpu))
                raise ValueError("Shoudl not adjust speedup here")
            slope = speedup_diff / cpu_diff
            # y = y1 + m(x-x1)
            # x1 = the modified cpu demand
            new_speedup = 1 + slope * (job.job_cpu_demand - fair_cpu)
                
            #self.logger.info("{}: orig={}, new={}, old_cpu={}, new_cpu={}".format(str(job), job.synergy_speedup, new_speedup, ideal_cpu, job.job_cpu_demand))
            job.synergy_speedup = new_speedup
        # Update job iteration time and num_iterations. Model carries per gpu iteration time
        # for fair share. 
        job.job_iteration_time = (job_model.iteration_time)
        job.synergy_speedup_orig = job.synergy_speedup
        #job.job_iteration_time = (job_model.iteration_time/job.job_gpu_demand)
         
        # For philly workloads alone, set #iterations
        # Num Iteration is fixed assuming consolidated fair-share job [1s per-iter dur]
        if job.iter_is_duration:
            job.job_total_iteration = int(job.job_total_iteration/job.job_iteration_time)
            

    def analyze_philly_trace(self):
        total_job_durations = DataSeries(
            ['gpu', 'time (hours)'],
            series_id_filter=(0, self.total_jobs),
            no_filter=True)
        
        total_gpu_demand = DataSeries(
            ['time (hours)', 'total gpu demand'],
            series_id_filter=(0, self.total_jobs),
            no_filter=True)
      
        total_jobs = DataSeries(
            ['time (hours)', 'total runnable jobs'],
            series_id_filter=(0, self.total_jobs),
            no_filter=True)
       


        arrival_time = DataSeries(
            ['gpu', 'time (hours)'],
            series_id_filter=(0, self.total_jobs),
            no_filter=True)
        
        gpu_demand = DataSeries(
            ['empty column', 'gpu demand'],
            series_id_filter=(0, self.total_jobs),
            no_filter=True)
        


        self.jobs.sort(key=lambda job: job.job_arrival_time)
        q = PriorityQueue()
        j_q = PriorityQueue()
        for job in self.jobs:
            total_job_durations.put(job.job_gpu_demand, job.job_duration, 0)
            arrival_time.put(job.job_gpu_demand, job.job_arrival_time, 0)
            gpu_demand.put(0, job.job_gpu_demand, 0)
            q.put(
                (job.job_arrival_time, 
                job.job_gpu_demand))
            j_q.put(
                (job.job_arrival_time, 
                1))
            q.put(
                (job.job_arrival_time + job.job_duration, 
                -job.job_gpu_demand))
            j_q.put(
                (job.job_arrival_time + job.job_duration, 
                -1))
        while not q.empty():
            (time, delta_gpu_demand) = q.get()
            (time_j, num_jobs) = j_q.get()
            total_gpu_demand.put_delta(time, delta_gpu_demand, 0)
            total_jobs.put_delta(time, num_jobs, 0)
        #total_gpu_demand.plot_step(path='./graphs-multi/')
        #total_gpu_demand.plot_step(path='./graphs-filtered-1-1.5k/')
        total_gpu_demand.plot_step(path='./graphs-filtered-try/')
        total_jobs.plot_step(path='./graphs-filtered-try/')
        #total_job_durations.plot_cdf(path='./graphs-multi/', prefix='duration')
        total_job_durations.plot_cdf(path='./graphs-filtered-try/', prefix='duration')
        #total_job_durations.plot_cdf(path='./graphs-filtered-1-1.5k/', prefix='duration')
        #arrival_time.plot_cdf(path='./graphs-filtered-1-1.5k/', prefix='arrival')
        arrival_time.plot_cdf(path='./graphs-filtered-try/', prefix='arrival')
        #arrival_time.plot_cdf(path='./graphs-multi/', prefix='arrival')
        #gpu_demand.plot_cdf(path='./graphs-multi/', prefix='gpu')
        gpu_demand.plot_cdf(path='./graphs-filtered-try/', prefix='gpu')
        #gpu_demand.plot_cdf(path='./graphs-filtered-1-1.5k/', prefix='gpu')

    def generate_next_job(self, last_job_arrival_time,  arrival=-1):
        if self.workload_type == "philly":
            job = self.jobs[self.job_id % self.total_jobs]
            job.job_id = self.job_id
            if self.philly_arrival:
                self.logger.info("Arrival = {:.2f}".format(job.job_arrival_time/3600))
                pass
            elif arrival >= 0:
                job.job_arrival_time = round(arrival+job.job_id*0.001, 3)
            else:
                inter_arrival_time = poisson_next_arrival_time(self.jobs_per_hour)
                job.job_arrival_time = last_job_arrival_time + inter_arrival_time
            if self.prioritize:
                if self.series_id_filter[0] <= job.job_id < self.series_id_filter[1]:
                    job.job_priority = 1 
        else:
            job_id = self.job_id
            tenant_id = 0
            inter_arrival_time = poisson_next_arrival_time(self.jobs_per_hour)
            job_arrival_time = last_job_arrival_time + inter_arrival_time

            job_iteration_time = 1
            job_total_iteration = get_gavel_like_iter()
            #job_total_iteration = get_total_iteration_exp(5000, 600000)
            #job_total_iteration = get_total_iteration(5000, 140000)


            #job_total_iteration = get_total_iteration(5000, 500000)
            #job_total_iteration = get_total_iteration(5000, 50000)
            #job_total_iteration = get_total_iteration(360, 1080)
            job_gpu_demand = get_job_gpu_demand() 
            job_packing_score = None
            job_placement_score = None
            synergy_res_matrix = None
            synergy_storage_matrix = None
            job = Job(
                    job_id, 
                    job_arrival_time,
                    job_iteration_time,
                    job_total_iteration,
                    job_gpu_demand,
                    job_packing_score,
                    job_placement_score,
                    synergy_res_matrix,
                    synergy_storage_matrix,
                    tenant_id,
                    iter_is_duration=True)
        self.job_id += 1

        job.job_task, job.job_class_id = self.model_zoo.get_job_class()       
 

        # Update job iter times and CPU, Mem profiles for the job
        # based on the chosen model
        # add placement score and packing score for the job
        self.add_synergy_profile(job)

       
        #self.logger.info("Job {}, class={}, task={}".format(job.job_id, job.job_class_id, job.job_task))
        return job

    def print_job_task_split(self):
        print("-"*50)
        self.model_zoo.print_task_splits()
        print("-"*50)

    def get_job_task_split(self):
        tasks = self.model_zoo.tasks
        return (list(task.runnable_jobs for task in tasks.values()), list(task.total_jobs for task in tasks.values()))

    def get_current_job_id(self):
        return self.job_id

    def online_workload(self, jobs_per_hour):
        num_hours = 24 * 4
        num_jobs = int(math.ceil(jobs_per_hour)) * num_hours
        tenant_id = 0
        job_ids = np.arange(num_jobs)
        job_arrival_times = [0]*num_jobs
        last_arrival_time = 0
        for i in job_ids:
            inter_arrival_time = poisson_next_arrival_time(jobs_per_hour)
            last_arrival_time = last_arrival_time + inter_arrival_time
            job_arrival_times[i] = last_arrival_time
        job_iteration_times = [5]*num_jobs
        job_total_iterations = [720]*num_jobs
        job_gpu_demands = [4]*num_jobs
        job_packing_scores = np.zeros((num_jobs, num_jobs))
        num_placement_options = 2
        job_placement_scores = np.ones((num_jobs, num_placement_options))

        jobs = []
        for job_id in job_ids:
            job = Job(
                    job_id, 
                    job_arrival_times[job_id],
                    job_iteration_times[job_id],
                    job_total_iterations[job_id],
                    job_gpu_demands[job_id],
                    job_packing_scores[job_id],
                    job_placement_scores[job_id],
                    None,
                    None,
                    tenant_id)
            jobs.append(job)
        return jobs

    def populate_from_trace(self, trace):
        # Trace file has <job_id, model_name, arrival_time, num_iters, gpu_demand>
        jobs = []
        with open(trace, 'r') as fr:
            for line in fr:
                job_stats = line.strip().split(',')
                job = Job(
                    int(job_stats[0]), 
                    float(job_stats[2]),
                    1,
                    int(job_stats[3]),
                    #1,
                    int(job_stats[4]),
                    None,
                    None,
                    None,
                    None,
                    0,
                    iter_is_duration=False)
                job.job_task, job.job_class_id = self.model_zoo.get_job_class_by_name(job_stats[1])
                self.add_synergy_profile(job)
                jobs.append(job)
        return jobs
 



    def default_workload(self, jobs=5):
        tenant_id = 0
        num_jobs = jobs
        job_ids = np.arange(num_jobs)
        job_arrival_times = [0]*num_jobs
        job_iteration_times = [1]*num_jobs
        #job_total_iterations = [1000, 2000 ,3000 ,3000 ,3000]
        #job_total_iterations = [1000, 2000 ,3000 ,3000]
        #job_total_iterations = [2000]
        job_total_iterations = [2000]*num_jobs
        job_gpu_demands = [1]*num_jobs
        #job_gpu_demands = [1]*num_jobs

        jobs = []
        for job_id in job_ids:
            job = Job(
                    job_id, 
                    job_arrival_times[job_id],
                    job_iteration_times[job_id],
                    job_total_iterations[job_id],
                    job_gpu_demands[job_id],
                    None,
                    None,
                    None,
                    None,
                    tenant_id,
                    iter_is_duration=True)
            jobs.append(job)
            job.job_task = TaskName.IMAGE
            job.job_class_id = 0  
            self.add_synergy_profile(job)
        return jobs

    def get_all_jobs(self):
        return self.jobs

    def get_num_jobs(self):
        return len(self.jobs)
