import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import numpy as np
from metrics.stats import DataSeries, DataSeriesCollection


"""
Collect and monitor utilization stats for all servers and aggregate utilization 
in the cluster
"""

class ClusterUtilization():
    def __init__(
        self,
        num_servers,
        name="util"):

        self.num_servers = num_servers

        # Plot each server utilization in a graph
        if self.num_servers > 1: 
            name_suffix = name + "_server" 
            self.gpu_util = {server_id:DataSeries(['time (hours)', 'GPU '+ name_suffix + str(server_id) +'(%)'], no_filter=True) for server_id in range(self.num_servers)}
            self.cpu_util = {server_id:DataSeries(['time (hours)', 'CPU '+ name_suffix + str(server_id) + '(%)'], no_filter=True) for server_id in range(self.num_servers)}
            self.mem_util = {server_id:DataSeries(['time (hours)', 'Mem ' + name_suffix + str(server_id) +'(%)'], no_filter=True) for server_id in range(self.num_servers)}

        self.aggregate_gpu = DataSeries(['time (hours)', 'Agg GPU ' + name + '(%)'], no_filter=True)
        self.aggregate_cpu = DataSeries(['time (hours)', 'Agg CPU ' + name + '(%)'], no_filter=True)
        self.aggregate_mem = DataSeries(['time (hours)', 'Agg Mem ' + name + '(%)'], no_filter=True)
       
    def put(self, util_map, time, job_id=0):
        assert(self.num_servers == len(util_map.keys()))
        aggregate = np.zeros(3)
        for server_id, util_list in util_map.items():
            #gpu, cpu, mem, sspeed, net is the order in util_list
            if self.num_servers > 1 :
                self.gpu_util[server_id].put(time, util_list[0], job_id)
                self.cpu_util[server_id].put(time, util_list[1], job_id)
                self.mem_util[server_id].put(time, util_list[2], job_id)
            aggregate[0] += util_list[0]
            aggregate[1] += util_list[1]
            aggregate[2] += util_list[2]

        # Average util across all servers in cluster
        aggregate = aggregate/self.num_servers
        self.put_aggregate(aggregate, time, job_id)

    def put_aggregate(self, agg_list, time, job_id=0):
        # agg list has agg_usage in oirder : gpu, cpu, mem, sspeed
        assert(len(agg_list) >= 3)
        self.aggregate_gpu.put(time, agg_list[0], job_id)
        self.aggregate_cpu.put(time, agg_list[1], job_id)
        self.aggregate_mem.put(time, agg_list[2], job_id)

    def plot_aggregate(self, path='./', stat=None):
        self.aggregate_gpu.plot_step(path=path, mean=True, metric="gpu-"+stat)
        self.aggregate_cpu.plot_step(path=path, mean=True, metric="cpu-"+stat)
        self.aggregate_mem.plot_step(path=path, mean=True)

    def plot_per_server(self, path="./"):
        if self.num_servers > 1:
            for server_id in self.gpu_util:
                self.gpu_util[server_id].plot_step(path=path, serv_id=server_id, metric="gpu")
                self.cpu_util[server_id].plot_step(path=path, serv_id=server_id, metric="cpu")
                self.mem_util[server_id].plot_step(path=path)
