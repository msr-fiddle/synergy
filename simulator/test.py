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

from resources.cluster import Cluster
from resources.server_config import CustomServerConfig
from resources.rack import Rack
from jobs.workload import Workload
from schedulers.fifo import FIFO
#from schedulers.srtf import SRTF
from schedulers.srtf_synergy_new import SRTF
from schedulers.srtf_synergy import SRTFSynergy
from schedulers.srtf_synergy_adjust import SRTFSynergyAdjust
from schedulers.fifo_synergy import FIFOSynergy
from schedulers.fifo_synergy_adjust import FIFOSynergyAdjust
from schedulers.las_synergy import LASSynergy
from schedulers.las_synergy_adjust import LASSynergyAdjust
from schedulers.las import LAS
from schedulers.srsf import SRSF
from schedulers.scheduler import Scheduler
from event_queue import EventQueue
from events.cluster_event import ClusterEvent
from events.job_arrival_event import JobArrivalEvent
from events.schedule_event import ScheduleEvent
from metrics.stats import DataSeries, DataSeriesCollection
from metrics.cluster_util import ClusterUtilization

class Simulator:

    def __init__(
        self,
        cluster_job_log,
        scheduler='SRTF',
        jobs_per_hour=5,
        series_id_filter=(0, 5),
        model_class_split=(34,33,33),
        exponential=True,
        multigpu=False,
        debug_multi=False,
        placement=True,
        prioritize=False,
        fair=True,
        tune=False):
        self.logger = logging.getLogger(__name__)
        ClusterEvent.simulator = self
        Scheduler.simulator = self

        if debug_multi:
            jobs_per_hour = 0.5
            #jobs_per_hour = 1
            #series_id_filter=(10, 12)
            series_id_filter=(100, 150)
            #series_id_filter=(0, 3)
            self.cluster = Cluster(config_file='configs/test_multidebug_config.ini')
        elif multigpu:
            jobs_per_hour = 1
            series_id_filter=(0, 3)
            self.cluster = Cluster(config_file='configs/test_multi_config.ini')
        else:
            self.cluster = Cluster(config_file='configs/test_config.ini')

        print(self.cluster.per_server_size)
        print(self.cluster.per_server_size_fair)

        self.time = 0
        self.time_limit = -1
        self.terminate = False
        self.runnable_jobs = list()
        self.finished_jobs = list()


        # Keep a log of max resoures in cluster to plot utilization
        _,self.max_servers, self.max_gpus, self.max_cpus,self.max_mem,self.max_sspeed, self.max_net=self.cluster.size
        self.logger.info("Cluster GPUs={}, CPUs={}, Mem={}GB, Sspeed={}MB/s".format(self.max_gpus, self.max_cpus, self.max_mem, self.max_sspeed))

        self.workload = Workload(
            cluster_job_log=cluster_job_log,
            jobs_per_hour=jobs_per_hour,
            exponential=exponential,
            prioritize=prioritize,
            multigpu=multigpu,
            debug_multi=debug_multi,
            series_id_filter=series_id_filter,
            model_class_split=model_class_split,
            per_server_size=self.cluster.per_server_size)
        self.scheduler = globals()[scheduler](placement=placement, fair=fair, tune=tune)
        
        self.event_queue = EventQueue()
        self.init_event_queue()

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
        

    def run_simulation(self):
        num_events=0
        while not self.event_queue.empty() and not self.terminate:
            event = self.event_queue.get()
            self.logger.debug(str(event))
            event.handleEvent()
            num_events += 1
        return self.get_stats()

    def get_stats(self):
        return (self.total_gpu_demand, 
                self.job_completion_times)

    def make_plots(self, dir_path):
        self.total_runnable_jobs.plot_step(path=dir_path)
        self.total_gpu_demand.plot_step(path=dir_path, must_print=True)
        self.job_completion_times.plot_cdf(path=dir_path)
        self.cluster_util.plot_aggregate(path=dir_path, stat="util")
        self.cluster_alloc.plot_aggregate(path=dir_path, stat="alloc")
        self.cluster_demand.plot_aggregate(path=dir_path, stat="demand")
        self.job_expected_duration.plot_cdf(path=dir_path, prefix="expected-duration")
        #self.cluster_util.plot_per_server(path=dir_path)
        #self.cluster_alloc.plot_per_server(path=dir_path)
        #self.cluster_demand.plot_per_server(path=dir_path)

    def add_event(self, event):
        self.event_queue.put(event)
    
    def add_next_job(self):
        job = self.workload.generate_next_job(self.time)
        self.add_event(JobArrivalEvent(job.job_arrival_time, job))
    
    def init_event_queue(self):
        self.add_next_job()
        self.add_event(ScheduleEvent(0, self.scheduler))

    def is_measurement_complete(self, job_id):
        if self.series_id_filter[0] <= job_id < self.series_id_filter[1]:
            self.filtered_ids += 1
        if self.filtered_ids ==\
            self.series_id_filter[1] - self.series_id_filter[0]:
            return True
        return False
    
    def get_time(self):
        return self.time
    
    def set_time(self, time):
        self.time = time
        if self.time_limit > 0 and self.time > self.time_limit:
            self.terminate = True

    def get_runnable_jobs(self):
        return self.runnable_jobs

    def start_job(self, job):
        self.runnable_jobs.append(job)
        self.total_runnable_jobs.put_delta(self.time, 1, job.job_id)
        self.workload.add_runnable_job(job.job_class_id)
        self.total_gpu_demand.put_delta(
            self.time, 
            job.job_gpu_demand*100./self.cluster.get_num_gpus(), 
            job.job_id)
        #if job.job_id % 1000 == 0:
        #    print("Allocation : ", self.cluster.alloc_stats)
        #    print("Utilization : ", self.cluster.utilization_stats)

    def finish_job(self, job):
        # self.logger.info("Finished Job: %s", str(job.job_id))
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
        self.logger.info("[{}] : Finished {} at {:.2f}hrs, arrival:{:.2f}hrs, iter={:.2f}s, num_iter={}, {}".format(job.job_id, job.job_model.model_name, self.time/3600, job.job_arrival_time/3600, job.job_iteration_time, job.job_total_iteration, str(job)))
        if self.is_measurement_complete(job.job_id):
            self.logger.info("Terminating workload at {:.2f} hrs : last job ID {}, total finished {}, pending {}".format(self.time/3600, job.job_id, len(self.finished_jobs),len(self.runnable_jobs)))
            self.terminate = True

def benchmark(seed, cluster_job_log, use_cache, cache_result, prioritize, plot=False, 
         exponential=True,  multigpu=False, debug_multi=False, placement=True, fair=True, tune=False):
    logger = logging.getLogger(__name__)

    #schedulers = ['LAS', 'LASSynergy']
    #schedulers = ['SRTF', 'SRTFSynergy',]
    #schedulers = ['SRTF', 'SRTFSynergy', 'FIFO', 'FIFOSynergy']
    #schedulers = ['FIFO', 'FIFOSynergy']
    schedulers = ['SRTF']
    #schedulers = ['LAS', 'LASSynergyAdjust']
    #schedulers = ['SRTF', 'SRTFSynergy', 'SRTFSynergyAdjust']
    #schedulers = ['FIFO', 'LAS', 'SRTF'] 
    jobs_per_hours = np.arange(1.0, 1.1,  0.2)
    class_split=[(33,33,33)]
    agg_total_gpu_demand_collection = DataSeriesCollection()
    agg_job_completion_times_collection = DataSeriesCollection()
    one_done = False 
    one_done_name = []
    for split in class_split:
        total_gpu_demand_collection = DataSeriesCollection()
        job_completion_times_collection = DataSeriesCollection()
        for scheduler in schedulers:
            #if one_done and 'Synergy' not in scheduler and scheduler in one_done_name:
                #continue
            if 'Synergy' not in scheduler and not one_done and scheduler not in one_done_name:
                one_done_name.append(scheduler)
                one_done = True
            for jobs_per_hour in jobs_per_hours:
                no_agg=False
                index = (scheduler, np.round(jobs_per_hour, 1), split)
                random.seed(seed)
                #if use_cache and one_done and scheduler in one_done_name and 'Synergy' not in scheduler:
                 #   index = (scheduler, np.round(jobs_per_hour, 1), (0,100,0))
                 #   no_agg=True
                print("Running {}".format(index))
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
                        #if not os.path.exists(dir_path):
                            #os.makedirs(dir_path)
                            #job_completion_times.plot_cdf(path=dir_path)
                            #total_gpu_demand.plot_step(path=dir_path)
                            #cluster_util.plot_aggregate(path=dir_path)
                            #cluster_alloc.plot_aggregate(path=dir_path)
                            #cluster_demand.plot_aggregate(path=dir_path)
                            #cluster_util.plot_per_server(path=dir_path)
                            #cluster_alloc.plot_per_server(path=dir_path)
                            #cluster_demand.plot_per_server(path=dir_path)

                else:
                    simulator = Simulator(
                        cluster_job_log=cluster_job_log,
                        scheduler=scheduler, 
                        jobs_per_hour=jobs_per_hour,
                        model_class_split=split,
                        exponential=exponential,
                        multigpu=multigpu,
                        debug_multi=debug_multi,
                        placement=placement,
                        prioritize=prioritize,
                        fair=fair,
                        tune=tune)
                    (total_gpu_demand, job_completion_times) = simulator.run_simulation()
                    if cache_result:
                        pickle.dump(
                            (total_gpu_demand, job_completion_times),
                            open("cache/stats_%s_%s_%s.pickle" % index, "wb"))
                        pickle.dump(
                            (simulator.cluster_util, simulator.cluster_alloc, simulator.cluster_demand),
                            open("cache/util_%s_%s_%s.pickle" % index, "wb"))

                    if plot:
                        dir_path = './graphs/per_run/%s_%s_%s' % index
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)
                        simulator.make_plots(dir_path)
                
                    (runnable_task_split, overall_task_split) = \
                        simulator.workload.get_job_task_split()
                    logger.info("Task split (runnable) : image={}, lang={}, speech={}".format(*runnable_task_split))
                    logger.info("Task split (overall) : image={}, lang={}, speech={}".format(*overall_task_split))
                #simulator.workload.print_job_task_split()

                #runnable = [i.job_id for i in simulator.runnable_jobs]
                #finished = [i.job_id for i in simulator.finished_jobs]
                #print(runnable)
                #print(finished)

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
   
     
        dir_path = './graphs/combined/%s_%s_%s' % split
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        job_completion_times_collection.plot_cdf()
        total_gpu_demand_collection.plot_weighted_mean(
            xlabel="Load (jobs/hour)", ylabel="Avg. GPU Demand (%)")
        job_completion_times_collection.plot_mean(
            xlabel="Load (jobs/hour)", ylabel="Avg. JCT (hours)")

        for f in os.listdir('./'):
            if f.endswith('.png'):
                shutil.move(f, os.path.join(dir_path,f))

    agg_job_completion_times_collection.plot_cdf()
    agg_total_gpu_demand_collection.plot_weighted_mean(
        xlabel="Load (jobs/hour)", ylabel="Avg. GPU Demand (%)")
    agg_job_completion_times_collection.plot_mean(
        xlabel="Load (jobs/hour)", ylabel="Avg. JCT (hours)")

def parser():
    parser = argparse.ArgumentParser(description='Parse Arguments.')
    
    parser.add_argument('--seed', default=42, type=int)

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
    # MultiGPU workload
    parser.add_argument('--multigpu', default=False, action="store_true")
    parser.add_argument('--debug_multi', default=False, action="store_true")
    parser.add_argument('--no_exp', default=False, action="store_true")
    parser.add_argument('--no_placement', default=False, action="store_true")
    parser.add_argument('--no_fair', default=False, action="store_true")
    parser.add_argument('--tune', default=False, action="store_true")

    # debug mode 
    parser.add_argument('--debug', default=False, action="store_true")
    args = parser.parse_args()
    return args


def debug(cluster_job_log):
    logger = logging.getLogger(__name__)

    scheduler = 'FIFO'
    jobs_per_hour = 5
    simulator = Simulator(
        cluster_job_log=cluster_job_log,
        scheduler=scheduler,
        jobs_per_hour=jobs_per_hour)
    (total_gpu_demand, job_completion_times) = simulator.run_simulation()

    

if __name__ == '__main__':
    args = parser()

    random.seed(args.seed)

    if args.debug:
        log_level = logging.DEBUG
    else:
        #log_level = logging.DEBUG
        log_level  = logging.INFO

    logging.basicConfig(
        format='%(module)s - %(funcName)s - %(levelname)s - %(message)s', 
        level=log_level)
   
    print("Running with placement={}, fair-share={}, tune={}, prioritize-all={}, multigpu={}, multidebug={}, exp={}"
            .format(not args.no_placement, not args.no_fair, args.tune, args.prioritize, args.multigpu
             , args.debug_multi, not args.no_exp)) 


    if not args.no_use_cache:
        if not os.path.exists('cache'):
            os.makedirs('./cache')   
    if args.debug_multi:
        args.multigpu=True


    if args.analyze_trace:
        # analyze philly trace
        workload = Workload(
            cluster_job_log=args.cluster_job_log, 
            sum_attempts=(not args.no_sum_attempts))
        workload.analyze_philly_trace()
    #elif args.debug:
    #    debug(None)
    else:
        # benchmark with increasing load
        benchmark(
            seed=args.seed,
            cluster_job_log=args.cluster_job_log, 
            use_cache=(not args.no_use_cache),
            cache_result=(not args.no_cache_result),
            prioritize=args.prioritize,
            exponential=(not args.no_exp),
            multigpu=args.multigpu,
            debug_multi=args.debug_multi,
            placement=(not args.no_placement),
            fair=(not args.no_fair),
            tune=args.tune,
            plot=args.plot)

    # todo:
    # > map a job to a known job in the lookup table for other metrics 
    # (nearest iteration time)
