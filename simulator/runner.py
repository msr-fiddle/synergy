import os
import sys
import random
import logging
import argparse
import pickle
import pandas as pd
import numpy as np
from queue import PriorityQueue
import shutil
import threading
import time
import copy
from atomic_update import AtomicUpdate
import math

#Cluster setup
from resources.cluster import Cluster
from resources.server_config import CustomServerConfig
from resources.rack import Rack
from jobs.workload import Workload

# Schedulers
from schedulers.fifo_synergy_new import FIFO
from schedulers.las_synergy_new import LAS
from schedulers.srtf_synergy_new import SRTF
from schedulers.ftf_synergy_new import FTF
from schedulers.drf import DRF
from schedulers.tetris import TETRIS
from schedulers.srsf import SRSF
from schedulers.scheduler import Scheduler

#Events
from event_queue import EventQueue
from events.cluster_event import ClusterEvent
from events.job_arrival_event import JobArrivalEvent
from events.schedule_event import ScheduleEvent
from events.allocation_event import AllocationEvent
from events.deploy_event import DeployEvent

#Graphs and metrics
from metrics.stats import DataSeries, DataSeriesCollection
from metrics.cluster_util import ClusterUtilization

#Runtime
from deployment.runtime.rpc import scheduler_server
from deployment.runtime.rpc import scheduler_client
import schedulers.callbacks as scheduler_callbacks
from deployment.helper import get_self_ip

SCHEDULER_PORT = 14000

class Runner:

    def __init__(
        self,
        cluster_job_log,
        scheduler='SRTF',
        jobs_per_hour=5,
        #series_id_filter=(15, 40),
        #series_id_filter=(1, 7999),
        #series_id_filter=(3000, 4000),
        series_id_filter=(4000, 5000),
        #series_id_filter=(25, 50),
        model_class_split=(34,33,33),
        exponential=True,
        philly_arrival=False,
        multigpu=False,
        small_trace=False,
        placement=True,
        prioritize=False,
        fair=True,
        tune=False,
        opt=False,
        simulate=True,
        round_duration=300,
        conn_list=None,
        config_file='configs/default_cluster.ini',
        num_jobs_default=0,
        static=False,
        record_trace=False,
        rec_trace_file=None,
        trace=None):

        self.logger = logging.getLogger(__name__)
        ClusterEvent.runner = self
        Scheduler.runner = self



        self.round_duration = round_duration
        self.num_jobs_default = num_jobs_default
        self.num_jobs_so_far = 0
        self.time = 0
        self.static = static 
        self.sched_port = SCHEDULER_PORT
        self.time_limit = -1
        self.terminate = False
        self.runnable_jobs = list()
        self.record_trace = record_trace
        self.rec_trace_file = rec_trace_file
        self.trace = trace
        if record_trace:
            self.rec_trace_file += str(jobs_per_hour)
            if fair:
                self.rec_trace_file += '_fair' 
            if tune:
                self.rec_trace_file += '_tune' 
            if opt:
                self.rec_trace_file += '_opt' 

            fw = open(self.rec_trace_file, 'w+')
        # IDs of jobs to be run in the next round
        self.job_ids_to_run = []
        self.job_ids_finished_this_round = []

        self.finished_jobs = list()
        self.real_start_time = time.time()
        self.sched_job_threshold = 0

        # IDs of jobs whose lease is renewed for the subsequent round.
        # Must be populated by the background scheduling process before the end
        # of the current scheduling round
        self.job_lease_status = dict()
        self.round_end_report = dict()

        self.simulate = simulate
        self.conn_list = conn_list
        self.config_file = config_file

        self.scheduler_lock = threading.Lock()
        self.done_sched_next_round = AtomicUpdate()
        self.ready_to_deploy_next_round = AtomicUpdate()
        self.deploy_ongoing = AtomicUpdate()


        self.cluster = Cluster(config_file=self.config_file, simulate=self.simulate, conn_list = self.conn_list)
        self.cluster_backup = None
        self.cluster_cleanstate = None

        # If launching obs on physical cluster, wait until all workers have been registered
        if not self.simulate:
            self.scheduler_callbacks = scheduler_callbacks.SchedulerCallbacks(self)
            callbacks = {
                'RegisterWorker': self.scheduler_callbacks.register_worker_callback,
                'RegisterJob' : self.scheduler_callbacks.register_job_callback,
                'UpdateIters' : self.scheduler_callbacks.update_iters_callback,
                'RoundEnd' : self.scheduler_callbacks.round_end_callback,
                'LeaseEnded' : self.scheduler_callbacks.lease_ended_callback
            }

            #Scheduler server
            self.server_thread = threading.Thread(target=scheduler_server.serve,
                                                  args=(SCHEDULER_PORT, callbacks))
            self.server_thread.daemon = True
            self.server_thread.start()

            # Asynchronous allocation thread
            #self.alloc_thread = threading.Thread(target=self.get_allocation)
            #self.alloc_thread.daemon = True
            #self.alloc_thread.start()

            while len(self.cluster.machine_to_rpc_map.keys()) != len(self.cluster.servers):
                time.sleep(0.5)

            self.cluster_cleanstate = copy.deepcopy(self.cluster)

        # Keep a log of max resoures in cluster to plot utilization
        _,self.max_servers, self.max_gpus, self.max_cpus,self.max_mem,self.max_sspeed, self.max_net=self.cluster.size
        self.logger.info("Cluster GPUs={}, CPUs={}, Mem={}GB, Sspeed={}MB/s".format(self.max_gpus, self.max_cpus, self.max_mem, self.max_sspeed))
        self.logger.info("Running {} with exp={}, multigpu={}, plcement={}, fair={}, tune={}, opt={}, prio-all={}"
           .format(scheduler, exponential,multigpu,placement,fair,tune,opt,prioritize))

        self.workload = Workload(
            cluster_job_log=cluster_job_log,
            jobs_per_hour=jobs_per_hour,
            exponential=exponential,
            philly_arrival=philly_arrival,
            prioritize=prioritize,
            multigpu=multigpu,
            small_trace=small_trace,
            series_id_filter=series_id_filter,
            model_class_split=model_class_split,
            per_server_size=self.cluster.per_server_size,
            num_jobs_default=self.num_jobs_default,
            trace=trace)
        self.scheduler = globals()[scheduler](
            round_duration=round_duration, 
            placement=placement, 
            fair=fair, 
            tune=tune,
            opt=opt,
            simulate=simulate)
        
        self.event_queue = EventQueue()
        
        self.init_event_queue(simulate=self.simulate)

        self.series_id_filter = series_id_filter
        self.filtered_ids = 0
        self.total_runnable_jobs = DataSeries(
            ['time (hours)', 'total jobs'],
            series_id_filter=series_id_filter,
            no_filter=True)
        self.total_gpu_demand = DataSeries(
            ['time (hours)', 'total gpu demand (%)'],
            series_id_filter=series_id_filter,
            no_filter=True)
        self.job_completion_times = DataSeries(
            ['job id', 'time (hours)'],
            series_id_filter=series_id_filter)
            #no_filter=True)
        self.job_expected_duration = DataSeries(
            ['job id', 'time (hours)'],
            series_id_filter=series_id_filter,
            no_filter=True)
        # Cluster statistics
        self.cluster_util = ClusterUtilization(self.max_servers, name="util")
        self.cluster_alloc = ClusterUtilization(self.max_servers, name="alloc")
        self.cluster_demand = ClusterUtilization(self.max_servers, name="demand")
        #print(self.cluster.alloc_stats)



    def remove_finished_jobs(self):
        for job_id in self.job_ids_finished_this_round:
            if job_id in self.job_ids_to_run:
                self.job_ids_to_run.remove(job_id)
        self.job_ids_finished_this_round = []

    def run_simulation(self):
        num_events=0
        while not self.event_queue.empty() and not self.terminate:
            event = self.event_queue.get()
            #if "SCHEDULE" not in str(event) and "LEASE" not in str(event):
            #    self.logger.info("Time: {:.2f}, Current Event: {}".format(self.get_time(), str(event)))
            #    self.logger.info("Time: {:.2f}, Queue: {}".format(self.get_time(), self.event_queue))
            event.handleEvent()
            num_events += 1
        return self.get_stats()

    def run_deployment(self):
        while not self.terminate:
            event = self.event_queue.get()
            if event.time > self.get_time():
                self.event_queue.put(event)
                time.sleep(0.5)
                continue
            self.logger.info("Time: {:.2f}, Current Event: {}".format(self.get_time(), str(event)))
            self.logger.info("Time: {:.2f}, Queue: {}".format(self.get_time(), self.event_queue))
            event.handleEvent()
            time.sleep(0.5)
        return self.get_stats()
            


    def get_stats(self):
        return (self.total_gpu_demand, 
                self.job_completion_times)

    def make_plots(self, dir_path):
        self.total_runnable_jobs.plot_step(path=dir_path)
        self.total_gpu_demand.plot_step(path=dir_path)
        self.job_completion_times.plot_cdf(path=dir_path)
        self.job_expected_duration.plot_cdf(path=dir_path, prefix="expected-duration")
        if self.simulate:
            self.cluster_util.plot_aggregate(path=dir_path, stat="util")
            self.cluster_alloc.plot_aggregate(path=dir_path, stat="alloc")
            self.cluster_demand.plot_aggregate(path=dir_path, stat="demand")
        #self.cluster_util.plot_per_server(path=dir_path)
        #self.cluster_alloc.plot_per_server(path=dir_path)
        #self.cluster_demand.plot_per_server(path=dir_path)

    def add_event(self, event):
        self.event_queue.put(event)
    
    def add_next_job(self, arrival=-1):
        job = self.workload.generate_next_job(self.get_time(), arrival=arrival)
        self.add_event(JobArrivalEvent(job.job_arrival_time, job))
        self.num_jobs_so_far += 1
        if self.record_trace:
           fw = open(self.rec_trace_file, 'a+')
           self.record_job(job, fw)
   
 
    def init_event_queue(self, simulate=True):
        if simulate:
            if self.trace is not None:
                arr_time  = 0
                for job in self.workload.jobs:
                    self.add_event(JobArrivalEvent(job.job_arrival_time, job))
                arr_time = math.ceil(self.workload.jobs[0].job_arrival_time /self.round_duration)* self.round_duration
                if self.static:
                    arr_time = 1
                self.add_event(ScheduleEvent(arr_time, self.scheduler))

            elif self.static:
                self.logger.info("Static workload")
                job = self.add_next_job(arrival=0)

                self.add_event(ScheduleEvent(1, self.scheduler))
            else:
                self.logger.info("Dynamic workload")
                self.add_next_job()
                self.add_event(ScheduleEvent(0, self.scheduler))
        else:
            #Add all jobs from workload file, static arrival
            for job in self.workload.jobs:
                self.add_event(JobArrivalEvent(job.job_arrival_time, job))
            self.add_event(AllocationEvent(self.get_time(), self.scheduler))
            self.add_event(DeployEvent(self.get_time(), self.scheduler))


    def record_job(self, job, fw):
        string   = "{},{},{},{},{}\n".format(job.job_id, job.job_model.model_name, job.job_arrival_time, job.job_total_iteration, job.job_gpu_demand)
        fw.write(string)


    def is_measurement_complete(self, job_id):
        if self.series_id_filter[0] <= job_id < self.series_id_filter[1]:
            self.filtered_ids += 1
        if self.filtered_ids ==\
            self.series_id_filter[1] - self.series_id_filter[0]:
            return True
        return False
    
    def get_time(self):
        if self.simulate:
            return self.time
        else:
            # Relative current time since the start of this workload
            return time.time() - self.real_start_time
    
    def set_time(self, time):
        self.time = time
        if self.time_limit > 0 and self.time > self.time_limit:
            self.terminate = True

    def get_runnable_jobs(self):
        return self.runnable_jobs

    def get_job_by_id(self, job_id):
        for job in self.runnable_jobs:
            if job.job_id == job_id:
                return job

    def start_job(self, job):
        self.logger.info("[{}] : Starting at {:.2f}s, arr = {:.2f}, dur = {:.2f}".format(str(job), (time.time()-self.real_start_time), job.job_arrival_time/3600, job.job_iteration_time*job.job_total_iteration))
        self.runnable_jobs.append(job)
        self.total_runnable_jobs.put_delta(self.time, 1, job.job_id)
        self.workload.add_runnable_job(job.job_class_id)
        self.total_gpu_demand.put_delta(
            self.time, 
            job.job_gpu_demand*100./self.cluster.get_num_gpus(), 
            job.job_id)

    def finish_job(self, job):
        #self.logger.info("[{}] : Finishing  at {:.2f}s".format(str(job), (time.time()-self.real_start_time)))
        self.job_ids_finished_this_round.append(job.job_id)
        self.runnable_jobs.remove(job)
        self.finished_jobs.append(job)
        self.workload.remove_runnable_job(job.job_class_id)
        self.total_runnable_jobs.put_delta(self.time, -1, job.job_id)
        self.job_expected_duration.put(
            job.job_id, 
            job.job_iteration_time*job.job_total_iteration, 
            job.job_id)
        self.total_gpu_demand.put_delta(
            self.time, 
            -job.job_gpu_demand*100./self.cluster.get_num_gpus(), 
            job.job_id)
        self.job_completion_times.put(
            job.job_id, 
            self.time - job.job_arrival_time,
            job.job_id)

        self.logger.info("[{}] : Finished {} at {:.2f}hrs, arrival:{:.2f}hrs, iter={:.2f}s, num_iter={}".format(job.job_id, job.job_model.model_name, self.time/3600, job.job_arrival_time/3600, job.job_iteration_time, job.job_total_iteration))
        #self.logger.info("[{}] : Finished {}:{} at {:.2f}hrs, arrival:{:.2f}hrs, iter={:.2f}s, num_iter={}".format(job.job_id, str(job), job.job_model.model_name, self.time/3600, job.job_arrival_time/3600, job.job_iteration_time, job.job_total_iteration))
        if self.is_measurement_complete(job.job_id):
            self.logger.info("Terminating workload at {:.2f} hrs : last job ID {}, total finished {}, pending {}".format(self.time/3600, job.job_id, len(self.finished_jobs),len(self.runnable_jobs)))
            self.terminate = True

def benchmark(seed, cluster_job_log, use_cache, cache_result, prioritize, plot=False, 
       exponential=True, philly_arrival=False, multigpu=False, debug_multi=False, placement=True, fair=True, tune=False, opt=False,
       simulate=True, conn_list=None, config_file=None, num_jobs_default=0, small_trace=False, static=False,
       record_trace=False, rec_trace=None, trace=None, plot_dir="./plots/"):
    logger = logging.getLogger(__name__)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Testing
    schedulers = ['FIFO+fair']
    scheduler_name = ['FIFO-Fair']

    # Intro
    #schedulers = ['LAS+fair' , 'LAS+tune', 'SRTF+fair', 'SRTF+tune']
    #scheduler_name = ['LAS-Fair', 'LAS-Tune', 'SRTF-Fair', 'SRTF-Tune']

    #schedulers = ['TETRIS', 'TETRIS+tune']
    #scheduler_name = ['TETRIS', 'TETRIS-tune']
    #schedulers = ['DRF', 'DRF+tune']
    #scheduler_name = ['DRF-Greedy', 'DRF-Tune']

    jobs_per_hours = np.arange(9.0, 10, 1)
    #jobs_per_hours = np.arange(0.5, 8.5, 0.5)
    
    class_split=[(20,70,10)]
    #class_split=[(20,70,10), (33,33,33), (50,0,50)]

    agg_total_gpu_demand_collection = DataSeriesCollection()
    agg_job_completion_times_collection = DataSeriesCollection()

    for split in class_split:
        total_gpu_demand_collection = DataSeriesCollection()
        job_completion_times_collection = DataSeriesCollection()
        for i, scheduler in enumerate(schedulers):
            # Fix place, fair, and tune based on scheduler
            placement = fair = tune = opt= False
            if 'place' in scheduler:
                placement = True
            if 'fair' in scheduler:
                fair = True
            if 'tune' in scheduler:
                tune = True
            if 'opt' in scheduler:
                opt = True

            for jobs_per_hour in jobs_per_hours:
                no_agg=False
                index = (scheduler_name[i], np.round(jobs_per_hour, 1), split)
                random.seed(seed)

                print("Running scheduler {} - {}".format(scheduler, index))
                scheduler = scheduler.split('+')[0]
                if use_cache and\
                   os.path.exists("cache/stats_%s_%s_%s.pickle" % index):
                    (total_gpu_demand, job_completion_times) = pickle.load(
                        open("cache/stats_%s_%s_%s.pickle" % index, "rb"))
                    fname = "cache/util_%s_%s_%s.pickle" % index
                    print(fname)
                    f = open(fname, "rb")
                    stats = pickle.load(f)
                    cluster_util, cluster_alloc, cluster_demand = stats

                    if plot:
                        dir_path = './graphs/per_run_cache/%s_%s_%s' % index
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)
                        job_completion_times.plot_cdf(path=dir_path)
                        total_gpu_demand.plot_step(path=dir_path)

                else:
                    runner = Runner(
                        cluster_job_log=cluster_job_log,
                        scheduler=scheduler, 
                        jobs_per_hour=jobs_per_hour,
                        model_class_split=split,
                        exponential=exponential,
                        philly_arrival=philly_arrival,
                        multigpu=multigpu,
                        small_trace=small_trace,
                        static=static,
                        placement=placement,
                        prioritize=prioritize,
                        fair=fair,
                        tune=tune,
                        opt=opt,
                        simulate=simulate,
                        conn_list=conn_list,
                        config_file=config_file,
                        num_jobs_default=num_jobs_default,
                        record_trace=record_trace,
                        rec_trace_file=rec_trace,
                        trace=trace)
                    if simulate:
                        (total_gpu_demand, job_completion_times) = runner.run_simulation()
                    else:
                        (total_gpu_demand, job_completion_times) = runner.run_deployment()
                    if cache_result:
                        pickle.dump(
                            (total_gpu_demand, job_completion_times),
                            open("cache/stats_%s_%s_%s.pickle" % index, "wb"))
                        pickle.dump(
                            (runner.cluster_util, runner.cluster_alloc, runner.cluster_demand),
                            open("cache/util_%s_%s_%s.pickle" % index, "wb"))

                    if plot:
                        dir_path = './graphs/per_run/%s_%s_%s' % index
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)
                        runner.make_plots(dir_path)
                
                    (runnable_task_split, overall_task_split) = \
                        runner.workload.get_job_task_split()
                    logger.info("Task split (runnable) : image={}, lang={}, speech={}".format(*runnable_task_split))
                    logger.info("Task split (overall) : image={}, lang={}, speech={}".format(*overall_task_split))


                total_gpu_demand_collection.put(
                    index, total_gpu_demand)
                job_completion_times_collection.put(
                    index, job_completion_times)
             
                if not no_agg:
                    agg_total_gpu_demand_collection.put(
                    index, total_gpu_demand)
                    agg_job_completion_times_collection.put(
                    index, job_completion_times)
                logger.info("{} : {}".format(scheduler, str(np.round(jobs_per_hour, 1))))
   
     

    agg_job_completion_times_collection.plot_cdf()
    agg_total_gpu_demand_collection.plot_weighted_mean(
        xlabel="Load (jobs/hour)", ylabel="Avg. GPU Demand (%)")
    agg_job_completion_times_collection.plot_mean(
        xlabel="Load (jobs/hour)", ylabel="Avg. JCT (hours)")

    cmd = "mv *.png " + plot_dir
    print("Moving plots to {}".format(plot_dir))
    os.system(cmd) 

def parser():
    parser = argparse.ArgumentParser(description='Parse Arguments.')
    
    parser.add_argument('--seed', default=42, type=int)

    # If not gicven cluster log, this is the number of jobs generated
    parser.add_argument('--num-jobs-default', default=0, type=int)
    # related to philly trace
    parser.add_argument('--cluster_job_log', default=None, type=str)
    # sum attempts duration by default
    parser.add_argument('--no_sum_attempts', default=False, action="store_true")
    # do not analyze trace by default
    parser.add_argument('--analyze_trace', default=False, action="store_true")
    # use cache by default
    parser.add_argument('--no_use_cache', default=False, action="store_true")
    # cache intermediate result by default
    parser.add_argument('--no_cache_result', default=False, action="store_true")
    # do not prioritize benchmarked jobs by default
    parser.add_argument('--prioritize', default=False, action="store_true")
    # Plot per-run micro stats
    parser.add_argument('--plot', default=False, action="store_true")
    parser.add_argument('--static', default=False, action="store_true")
    parser.add_argument('--small_trace', default=False, action="store_true")
    parser.add_argument('--no_exp', default=False, action="store_true")
    parser.add_argument('--multigpu', default=False, action="store_true")
    parser.add_argument('--no_placement', default=False, action="store_true")
    parser.add_argument('--no_fair', default=False, action="store_true")
    parser.add_argument('--tune', default=False, action="store_true")
    parser.add_argument('--philly_arrival', default=False, action="store_true")
    parser.add_argument('--opt', default=False, action="store_true")
    parser.add_argument('--record_trace', default=False, action="store_true")
    parser.add_argument('--rec_trace', default='./record', type=str)
    parser.add_argument('--replay_trace', default=None, type=str)
    parser.add_argument('--plot_dir', default="./plots/", type=str)
    parser.add_argument('--config_file', default='configs/default_cluster.ini', type=str)
    parser.add_argument('--conn_file', default=None, type=str)
    parser.add_argument('--no_simulate', default=False, action="store_true")
    # debug mode 
    parser.add_argument('--debug', default=False, action="store_true")
    args = parser.parse_args()
    return args


def debug(cluster_job_log):
    logger = logging.getLogger(__name__)

    scheduler = 'FIFO'
    jobs_per_hour = 5
    simulator = Runner(
        cluster_job_log=cluster_job_log,
        scheduler=scheduler,
        simulate=True,
        jobs_per_hour=jobs_per_hour)
    (total_gpu_demand, job_completion_times) = simulator.run_simulation()

    

if __name__ == '__main__':
    args = parser()

    random.seed(args.seed)

    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level  = logging.INFO

    logging.basicConfig(
        format='%(module)s - %(funcName)s - %(levelname)s - %(message)s', 
        level=log_level)

    if not args.no_use_cache:
        if not os.path.exists('cache'):
            os.makedirs('./cache')   

    if args.analyze_trace:
        # analyze philly trace
        workload = Workload(
            cluster_job_log=args.cluster_job_log, 
            sum_attempts=(not args.no_sum_attempts),
            multigpu = args.multigpu)
        workload.analyze_philly_trace()
    elif args.debug:
        debug(None)
    else:
        # benchmark with increasing load
        benchmark(
            seed=args.seed,
            cluster_job_log=args.cluster_job_log, 
            use_cache=(not args.no_use_cache),
            cache_result=(not args.no_cache_result),
            prioritize=args.prioritize,
            exponential=(not args.no_exp),
            philly_arrival=(args.philly_arrival),
            multigpu=args.multigpu,
            small_trace=args.small_trace,
            static=args.static,
            placement=(not args.no_placement),
            fair=(not args.no_fair),
            tune=args.tune,
            opt=args.opt,
            simulate=(not args.no_simulate),
            conn_list=args.conn_file,
            config_file=args.config_file,
            plot=args.plot,
            num_jobs_default=args.num_jobs_default,
            record_trace=args.record_trace,
            rec_trace=args.rec_trace,
            trace=args.replay_trace,
            plot_dir=args.plot_dir)

