from enum import Enum
import heapq
import logging
from collections import OrderedDict
import copy
import sys
import numpy as np
import os
from resources.gpu import GPU
from resources.rack import Rack
from resources.server_config import DefaultServerConfig
from resources.server_config import CustomServerConfig
from helpers.utils import gpu_normalized_vector,  cumulative_map
import collections
import math

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0

def nested_update(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, collections.Mapping):
            tmp = nested_update(orig_dict.get(key, { }), val)
            orig_dict[key] = tmp
        elif isinstance(val, list):
            orig_dict[key] = (orig_dict.get(key, []) + val)
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict


class AllocationStrategy(Enum):
    DEFAULT_ORDER=1
    PLACEMENT_SENSITIVE=2
    SYNERGY_RANDOM=3
    SYNERGY_PLACEMENT=4

class Cluster():

    def __init__(self, 
                 config_file='configs/default_cluster.ini',
                 simulate=True,
                 conn_list=None):
        self.logger = logging.getLogger(__name__)
        self.racks = []

        self.simulate = simulate
        # Used only if not in simulation mode. File with list of worker IP, port
        self.conn_list = conn_list
        self.machine_to_rpc_map = dict()

        #Indexed by server ID. Contains server handle
        self.servers = []

        self.gpus = []
        self.config_map = dict()
        self.total_cpus=0
        self.total_sspeed=0
        self.total_net=0
        self.total_mem=0            

        self.populate_default_config_map()
        self.racks = self.build_from_config(config_file, conn_list=self.conn_list)
        self.init_cluster()

        total_list = [len(self.gpus), self.total_cpus, self.total_mem, self.total_sspeed, self.total_net]
        self.per_server_size =  [int(item/len(self.servers)) for item in  total_list]
        self.per_server_size_fair =  [item/self.per_server_size[0] for item in self.per_server_size ]

        self.server_free_map = OrderedDict()
        for i, serv in enumerate(self.servers):
            self.server_free_map[serv.id] = list(serv.availability_stats())

        # List of jobs scheduled for each server
        # Must be reset at the beginning of each round in scheduler
        self.server_job_schedule = [list() for server in self.servers]

    def init_cluster(self):
        for rack in self.racks:
            for _server in rack.server_list:
                self.servers.append(_server)
                self.total_cpus += _server.num_cpus
                self.total_mem += _server.memory
                self.total_sspeed += _server.storage_speed
                self.total_net += _server.network
            for _gpu in rack.gpu_list:
                self.gpus.append(_gpu)

        for server in self.servers:
            self.logger.debug("Server [{}] : {} GPUs ({})," \
                  "{} CPUs, {} GB memory, {} MB/s storage, {} Gbps Ethernet".format(
                   server.id, server.gpu_available, server.gpu_ids, server.cpu_available, server.mem_available, 
                   server.sspeed_available, server.net_available))

        self.logger.info("Built a cluster of {} GPUs across {} servers, each with {} GPUs, {} CPUs, {}GB DRAM, {}MB/s storage bandwidth and {}Gbps ethernet".format(len(self.gpus), len(self.servers), self.servers[0].gpu_available, self.servers[0].cpu_available, self.servers[0].mem_available, self.servers[0].sspeed_available, self.servers[0].net_available))
        

    def connection_to_server(self, ip, port):
        for _server in self.servers:
            self.logger.info("Server ID {} : Port:{}, ip={}, exp_port={} exp_ip={}".format( 
                _server.server_id, _server.port, _server.ip, port, ip))
            if port == _server.port and ip == _server.ip:
                self.logger.info("Returning {}".format(_server))
                return _server
        return None


    @property
    def size(self):
        # racks, servers, gpus, cpus, mem, sspeed, net
        return (len(self.racks), len(self.servers), len(self.gpus), self.total_cpus, self.total_mem, self.total_sspeed, self.total_net) 
   

    # Get cluster alloc statistics as a map
    # indexed by server ID - for each server, #GPU, #CPU, mem, sspeed, network that is alloc
    @property
    def alloc_stats(self):
        alloc = {}
        _, server, gpu, cpu, mem, sspeed, net = self.size
        for _server in self.servers:
            alloc[_server.id] = list(_server.alloc_stats(percent=True))
        return alloc
            
            
    # Get cluster utilization statistics as a map
    # indexed by server ID - for each server, #GPU, #CPU, mem, sspeed, network that is alloc
    @property
    def utilization_stats(self):
        util = {}
        for _server in self.servers:
            util[_server.id] = list(_server.utilization_stats(percent=True))
        #print("Util : ", util)
        return util

    # Get cluster demand statistics as a map
    # indexed by server ID - for each server, #GPU, #CPU, mem, sspeed, network that is alloc
    @property
    def demand_stats(self):
        demand = {}
        for _server in self.servers:
            demand[_server.id] = list(_server.demand_stats(percent=True))
        #print("Demand : ", demand)
        return demand
 
    def build_from_config(self, config_file, conn_list=None):
        racks = []
        if os.path.exists(config_file):
            self.logger.debug("Building cluster from {}".format(config_file))
            config = ConfigParser()
            config.read(config_file)
            self.build_config_map(config)
        else:
            self.logger.debug("Building default cluster...")
            self.populate_default_config_map()

        self.logger.info(self.config_map) 
        racks = []
        #self.logger.info("Built cluster of {} GPUs and {} servers each with CPUs-{}, sspeed-{}MB/s, gpus-{}, dram-{}GB".format(len(self.gpus), len(self.servers),self.config_map.get('CLUSTER').get('cpus_per_server'), self.config_map.get('CLUSTER').get('sspeed_per_server'), self.config_map.get('CLUSTER').get('gpus_per_server'),self.config_map.get('CLUSTER').get('dram_per_server')))
        server_config = CustomServerConfig(
          num_cpus=self.config_map.get('CLUSTER').get('cpus_per_server'),
          storage_speed=self.config_map.get('CLUSTER').get('sspeed_per_server'), 
          num_gpus=self.config_map.get('CLUSTER').get('gpus_per_server'),
          network=self.config_map.get('CLUSTER').get('net_per_server'),
          memory=self.config_map.get('CLUSTER').get('dram_per_server'))
            
        servers_per_rack = self.config_map.get('CLUSTER').get('servers_per_rack')
        for rack_id in range(self.config_map.get('CLUSTER').get('racks')):
            rack = Rack(rack_id, servers_per_rack, server_config=server_config, conn_list=conn_list)
            racks.append(rack)
        return racks
          

    def populate_default_config_map(self):
        self.config_map['CLUSTER'] = {}             
        self.config_map['CLUSTER']['racks'] = 1             
        self.config_map['CLUSTER']['servers_per_rack'] = 4       
        default_server = DefaultServerConfig()
        self.config_map['CLUSTER']['gpus_per_server'] = default_server.num_gpus  
        self.config_map['CLUSTER']['cpus_per_server'] = default_server.num_cpus
        self.config_map['CLUSTER']['dram_per_server'] = default_server.memory
        self.config_map['CLUSTER']['sspeed_per_server'] = default_server.storage_speed
        self.config_map['CLUSTER']['net_per_server'] = default_server.network
        del default_server
        
        self.config_map['SCHEDULER'] = {}             
        self.config_map['SCHEDULER']['policy'] = 'FIFO'
        self.config_map['SCHEDULER']['lease_time'] = 5
     

 
    def build_config_map(self, config):
        cmap = {}
        for section in config.sections():
            cmap[section] = {}
            for option in config.options(section):
                if 'policy' not in option:
                    cmap[section][option] = config.getint(section, option)
                else:
                    cmap[section][option] = config.get(section, option)
        self.config_map = nested_update(self.config_map, cmap)

    def default_cluster(self):
        job_id = -1
        tenant_id = -1
        num_racks = 1
        num_machines_per_rack = 27
        num_gpus_per_machine = 4
        
        rack_ids = np.arange(num_racks)
        machine_ids = dict()
        gpu_ids = dict()
        for rack_id in rack_ids:
            rack_machine_ids = np.arange(
                rack_id * num_machines_per_rack, 
                (rack_id + 1) * num_machines_per_rack)
            machine_ids[rack_id] = rack_machine_ids
            for machine_id in rack_machine_ids:
                machine_gpu_ids = np.arange(
                    machine_id * num_gpus_per_machine,
                    (machine_id + 1) * num_gpus_per_machine)
                gpu_ids[machine_id] = machine_gpu_ids
        
        gpus = []
        for rack_id in rack_ids:
            rack_machine_ids = machine_ids[rack_id]
            for machine_id in rack_machine_ids:
                machine_gpu_ids = gpu_ids[machine_id]
                for gpu_id in machine_gpu_ids:
                    gpu = GPU(rack_id, machine_id, gpu_id, job_id, tenant_id)
                    gpus.append(gpu)
        return gpus

    def get_all_gpus(self):
        return self.gpus

    def get_num_gpus(self):
        return len(self.gpus)

    def get_free_gpus(self, gpus=None):
        all_gpus = self.get_all_gpus()
        if gpus is not None:
            all_gpus = gpus
        _gpus = [gpu for gpu in all_gpus if gpu.is_free()]
        return _gpus

    def is_free(self, gpu):
        return gpu.is_free()


    # returns a map of server
    # to resource allocations in that server for the job
    def get_allocation_map(self, gpus):
        res_map = {}
        ret = {}
        if isinstance(gpus, list): 
            for _gpu in gpus:
                server_id = _gpu.machine_id
                if server_id not in ret:
                    ret[server_id] = 1
                else:
                    ret[server_id] += 1
        else:
            ret[gpus.machine_id] = 1

        for server in ret.keys():
            used_gpus = ret[server]
            _server = self.servers[server]
            fair_share_percent = used_gpus/_server.num_gpus
            res_map[_server] = {}
            res_map[_server]['gpu'] = used_gpus
            res_map[_server]['cpu'] = math.floor(_server.num_cpus*fair_share_percent)
            res_map[_server]['mem'] = _server.memory*fair_share_percent
            res_map[_server]['sspeed'] = _server.storage_speed*fair_share_percent

        del ret
        return res_map


    def allocate(self, available_gpus, num_gpus, job, time=None,
        alloc_strategy=AllocationStrategy.DEFAULT_ORDER, fair=True, tune=False):
        """
        Allocate one or more GPUs to a job from the given list of available GPUs.
        Return a list of remaining available GPUs after this allocation.
        """
        self.logger.debug("Allocation strategy = {}".format(alloc_strategy))
        if tune:
            return self.allocate_synergy_tune(available_gpus, num_gpus, job, time, alloc=alloc_strategy, fair=fair)
        elif alloc_strategy == AllocationStrategy.DEFAULT_ORDER:
            return self.allocate_default_order(available_gpus, num_gpus, job, time)
        elif alloc_strategy == AllocationStrategy.PLACEMENT_SENSITIVE:
            return self.allocate_placement_sensitive(available_gpus, num_gpus, job, time)
        elif alloc_strategy == AllocationStrategy.SYNERGY_RANDOM:
            return self.allocate_synergy_random(available_gpus, num_gpus, job, time, fair=fair)
        elif alloc_strategy == AllocationStrategy.SYNERGY_PLACEMENT:
            return self.allocate_synergy_placement(available_gpus, num_gpus, job, time, fair=fair)
        else:
            raise ValueError("Unrecognized allocation strategy: %s" % alloc_strategy)

    """
    Allocate [gpus] to the job for a duration <time>
    """
    def _do_allocate(self, gpus, job, time=None, res_map=None, fair=True):
        # Res_map is the resource alloc map for the job 
        # If none, cpu, mem are fair-shared wrt #gpus allotted per server
        # To specify allocation, res_map must be of the form
        # dict{server}{cpu, mem, gpu}
        if res_map is None:
            res_map = {}
            res_map = self.get_allocation_map(gpus)

        job.allocate(gpus, time, res_map=res_map, fair=fair, simulate=self.simulate)
        try:
            iter(gpus)
            for gpu in gpus:
                gpu.allocate(job)
                self.logger.debug("%s - %s", gpu, job)
        except TypeError:
            gpu.allocate(job)
            self.logger.debug("%s - %s", gpu, job)

        #update server info
        for _server in res_map.keys():
            _server.allocate(res_map[_server])

        # Update resource util and demand for the job
        job.update_utilization()


    def allocate_default_order(self, available_gpus, num_gpus, job, time=None):
        """
        Allocate the first N GPUs in the given list.
        Return a list of remaining available GPUs after this allocation.
        """
        self._do_allocate(available_gpus[:num_gpus], job, time)
        return available_gpus[num_gpus:]

    def allocate_placement_sensitive(self, available_gpus, num_gpus, job, time=None):
        """
        Allocate GPUs according to a job's placement preference.

        If a job prefers consolidation and there are machines with enough available
        GPUs, then we will allocate from the machine with the fewest number of available
        GPUs so as to minimize fragmentation. Otherwise, we will allocate 1 GPU at a
        time starting from machines with the fewest number of available GPUs.

        Return a list of remaining available GPUs after this allocation.
        """
        # Map from machine_id to a list of available GPUs on the machine
        machine_to_available_gpus = {}
        for gpu in available_gpus:
            if gpu.machine_id not in machine_to_available_gpus:
                machine_to_available_gpus[gpu.machine_id] = []
            machine_to_available_gpus[gpu.machine_id].append(gpu)
        # If the job prefers consolidation, just find a machine with enough GPUs
        # If no such machine exists, ignore this preference
        if job.prefers_consolidation():
            candidates = [m for m, gpus in machine_to_available_gpus.items()\
                if len(gpus) >= num_gpus]
            if len(candidates) > 0:
                target = sorted(candidates,\
                    key=lambda m: len(machine_to_available_gpus[m]))[0]
                gpus_to_allocate = machine_to_available_gpus[target][:num_gpus]
                self._do_allocate(gpus_to_allocate, job, time)
                return [g for g in available_gpus if g not in gpus_to_allocate]
        # Else, allocate from the machine with the fewest number of available GPUs
        # to minimize fragmentation
        pq = [(len(gpus), m) for m, gpus in machine_to_available_gpus.items()]
        heapq.heapify(pq)
        gpus_to_allocate = []
        while len(gpus_to_allocate) < num_gpus and len(pq) > 0:
            _, m = heapq.heappop(pq)
            gpus = machine_to_available_gpus[m]
            gpus_to_allocate.append(gpus.pop())
            if len(gpus) > 0:
                heapq.heappush(pq, (len(gpus), m))
        self._do_allocate(gpus_to_allocate, job, time)
        return [g for g in available_gpus if g not in gpus_to_allocate]


    def allocate_synergy_placement(self, available_gpus, num_gpus, job, time=None, fair=True, demand_vec=None):
        if fair:
            job_demand_vector = [res*num_gpus for res in self.per_server_size_fair]
        else:
            job_demand_vector = job.get_job_demand_vector

        if demand_vec is not None:
            job_demand_vector = demand_vec
 
        if num_gpus != job_demand_vector[0]:
            raise ValueError("Mismatch in GPU requirements")


        server_handle_to_available_gpus = {}
        for gpu in available_gpus:
            if gpu.server_handle not in server_handle_to_available_gpus:
                server_handle_to_available_gpus[gpu.server_handle] = []
            server_handle_to_available_gpus[gpu.server_handle].append(gpu)
        # If the job prefers consolidation, just find a machine with enough GPUs
        #If job prefers cnsolidation, find a server that fits the job with its requirements as is
        if job.prefers_consolidation():
            candidates = [s for s, gpus in server_handle_to_available_gpus.items()\
                if self._fits_in_server(None, job_demand_vector, server=s)]
            if len(candidates) > 0:
                target = sorted(candidates,\
                    key=lambda m: len(server_handle_to_available_gpus[m]))[0]
                gpus_to_allocate = server_handle_to_available_gpus[target][:num_gpus]
                # all these gpus are in a server
                alloc_map = self._build_alloc_map(job_demand_vector)
                res_map = {}
                res_map[target] = alloc_map

                self._do_allocate(gpus_to_allocate, job, time, res_map=res_map, fair=fair)
                servers_used = job.server_ids
                for serv in servers_used():
                    self.server_job_schedule[serv].append(job)
                return True, [g for g in available_gpus if g not in gpus_to_allocate]

        # If cannot be consolidated or does not prefer one, allocate from machines with least GPU
        # TODO: Should we prioritize wrt other resource fragmentation?
        job_demand_vector_gpu_norm = gpu_normalized_vector(job_demand_vector)
        gpus_to_allocate, res_map =  self._top_synergy_gpus_placement(job_demand_vector_gpu_norm, num_gpus, available_gpus)
        if gpus_to_allocate is None:
            return False, available_gpus
        
        self._do_allocate(gpus_to_allocate, job, time, res_map=res_map, fair=fair)
        servers_used = job.server_ids
        for serv in servers_used():
            self.server_job_schedule[serv].append(job)
        return True, [g for g in available_gpus if g not in gpus_to_allocate]           



    def allocate_synergy_random(self, available_gpus, num_gpus, job, time=None, fair=True, demand_vec=None):
        """
        Used by synergy skip policy
        Allocate first N GPUs to a job such that it satisfies job res demands
        Does not take into account consolidation of GPUs for a job
        Returns allocation status <success|failure> and list of free gpus

        Find per-GPU share vector and the top n GPUs that satsfy this res share 
        irrespective of the server ID [i.e. irrespective of consolidation]
        """

        if fair:
            job_demand_vector = [res*num_gpus for res in self.per_server_size_fair]
        else:
            job_demand_vector = job.get_job_demand_vector

        if demand_vec is not None:
            job_demand_vector = demand_vec
 
        if num_gpus != job_demand_vector[0]:
            raise ValueError("Mismatch in GPU requirements")

        job_demand_vector_gpu_norm = gpu_normalized_vector(job_demand_vector)
        gpus_to_allocate, res_map =  self._top_synergy_gpus(job_demand_vector_gpu_norm, num_gpus, available_gpus)
        if gpus_to_allocate is None:
            self.logger.debug("job_demand_vec={}, gpus_to_alloc={}".format(job_demand_vector_gpu_norm, gpus_to_allocate))
            return False, available_gpus
        
        self._do_allocate(gpus_to_allocate, job, time, res_map=res_map, fair=fair)
        servers_used = job.server_ids()
           
        for serv in servers_used:
            self.server_job_schedule[serv].append(job)


        self.logger.debug("Allocted job {} at {}".format(str(job), servers_used))
        for gpu in job.gpus:
            self.logger.debug("   {} ".format(str(gpu)))
        for server in servers_used:
            self.logger.debug("   {}:{} ".format(server, self.servers[server].availability_stats()))

        return True, [g for g in available_gpus if g not in gpus_to_allocate]           



    def _top_synergy_gpus_placement(self, norm_demand_vector, num_gpus, available_gpus): 
        """
         Returns gpus and an allocation map given the per-GPU normalized demand vector and #GPUs
         Temporarily holds resources at servers to recursively check requirements
         Allocates from m/c with least free GPUs to reduce fragmentation
        """
        gpus_to_allocate = []   
        res_map = {}
        
        server_handle_to_available_gpus = {}
        for gpu in available_gpus:
            if gpu.server_handle not in server_handle_to_available_gpus:
                server_handle_to_available_gpus[gpu.server_handle] = []
            server_handle_to_available_gpus[gpu.server_handle].append(gpu)

        pq = [(len(gpus), m) for m, gpus in server_handle_to_available_gpus.items()]
        heapq.heapify(pq)

        while len(gpus_to_allocate) < num_gpus and len(pq) > 0:
            _, server_handle = heapq.heappop(pq)
            if self._fits_in_server(None, norm_demand_vector, server_handle):
               gpus = server_handle_to_available_gpus[server_handle]
               server_handle.hold_resources(norm_demand_vector)
               server_alloc_map = self._build_alloc_map(norm_demand_vector)
               gpus_to_allocate.append(gpus.pop())
               if server_handle not in res_map.keys():
                    res_map[server_handle] =  server_alloc_map
               else:
                    res_map[server_handle] =  cumulative_map(res_map[server_handle], server_alloc_map)
               if len(gpus) > 0:
                   heapq.heappush(pq, (len(gpus), server_handle))

        for serv in self.servers:
            serv.release_held_resources()

        if len(gpus_to_allocate) < num_gpus:
            return None, None

        return gpus_to_allocate, res_map


    def _top_synergy_gpus(self, norm_demand_vector, num_gpus, available_gpus): 
        """
         Returns gpus and an allocation map given the per-GPU normalized demand vector and #GPUs
         Temporarily holds resources at servers to recursively check requirements
        """
        gpus_to_allocate = []   
        res_map = {}

        for gpu in available_gpus:
            if self._fits_in_server(gpu, norm_demand_vector):
                gpus_to_allocate.append(gpu)
                server_handle = gpu.server_handle
                server_handle.hold_resources(norm_demand_vector)
                server_alloc_map = self._build_alloc_map(norm_demand_vector)
                if server_handle not in res_map.keys():
                    res_map[server_handle] =  server_alloc_map
                else:
                    res_map[server_handle] =  cumulative_map(res_map[server_handle], server_alloc_map)

            if len(gpus_to_allocate) ==  num_gpus:
                # Release resouces held during this process
                for serv in self.servers:
                    serv.release_held_resources()
                return gpus_to_allocate, res_map
                    
        for serv in self.servers:
            serv.release_held_resources()
        return None, None

    def _fits_in_server(self, gpu, norm_demand_vector, server=None):
        if gpu is not None:
            free_vector = list(gpu.server_handle.availability_stats_with_hold())
        elif server is not None:
            free_vector = list(server.availability_stats_with_hold())
        else:
            return False

        for idx, free_res in enumerate(free_vector):
            required_res = norm_demand_vector[idx]
            if free_res < required_res:
                return False
        return True
            
            

    def _build_alloc_map(self, job_demand_vector):
        alloc_map = {}
        alloc_map["gpu"] = job_demand_vector[0]
        alloc_map["cpu"] = job_demand_vector[1]
        alloc_map["mem"] = job_demand_vector[2]
        alloc_map["sspeed"] = job_demand_vector[3]
        return alloc_map


    def allocate_synergy_tune(self, available_gpus, num_gpus, job, time, alloc=AllocationStrategy.SYNERGY_RANDOM, fair=True):
        if alloc == AllocationStrategy.SYNERGY_RANDOM:
            placement = False 
            _call_allocate = self.allocate_synergy_random
        else:
            placement =  True
            _call_allocate = self.allocate_synergy_placement

        job_demand_vector = job.get_job_demand_vector

        available_gpus = self._tune(job, job_demand_vector, num_gpus, False, True, False, _call_allocate, available_gpus, time, fair)
        return True, available_gpus


    def _tune(self, job, demand_vec, job_gpu_deficit, peer_adjust, initial, final, _call_allocate, available_gpus, time, fair):
        success, available_gpus = _call_allocate(available_gpus, job_gpu_deficit, job, time=time, fair=fair, demand_vec=demand_vec)
        if success:
            if final:
               self.logger.debug("Allocated {} with final={}".format(str(job), final))
            return available_gpus
        if final:
            self.logger.info("FAILURE for job {}, free_gpus={}".format(str(job), len(available_gpus)))
            for gpu in available_gpus:
                self.logger.info(gpu)
            for i, job_list in enumerate(self.server_job_schedule):
                for j in job_list:
            #for job in self.sver_job_schedule[gpu.machine_id]:
                    self.logger.info("Server {} : {}".format(i, str(j)))
                    for gpu in j.gpus:
                        self.logger.info("   {} ".format(str(gpu)))
                self.logger.info("Server {} : {}".format(i,self.servers[i].availability_stats()))

            self.logger.info(gpu.server_handle.availability_stats())
            sys.exit(1)

        #We could not allocate with the job's demands.
        # Switch job to fair-share
        self.logger.debug("Must tune job {}, initial={}, peer_adj={}, final={}".format(str(job), initial,peer_adjust,final))
        can_adjust, new_demand_vec = self._make_fair_share(job, demand_vec)
        job_gpu_deficit = job.get_gpu_deficit()

        if initial:
            if can_adjust:
                demand_vec = new_demand_vec
            return self._tune( job,demand_vec, job_gpu_deficit, False, False, False, _call_allocate, available_gpus, time, fair)
        elif not can_adjust and not peer_adjust:
            # Current job's demands amade fair-share already
            # Peer has not been adjusted yet
            return self._tune(job,demand_vec, job_gpu_deficit, True, False, False, _call_allocate, available_gpus, time, fair)
        elif peer_adjust and not final:
            # Get servers with underutilized GPU (randomly)
            server_handle_map = self._get_underutilized_servers(job_gpu_deficit, available_gpus, consolidate=job.prefers_consolidation())
            self.logger.debug("Underutil server map = {}".format(server_handle_map))
            #if server_id < 0:
            #    raise Exception("Invalid server")
            free_vec_map = {}
            for serv, gpus in server_handle_map.items():
                free_vec = list(serv.availability_stats())
                ratio = gpus/job_gpu_deficit
                demand_vec_share = [res*ratio for res in demand_vec]
                jobs_to_realloc = self._reallocate_peer(demand_vec_share, free_vec, serv)

                for j in jobs_to_realloc:
                    gpus_realloc = j.gpus
                    peer_res_map = {}
                    for serv in j.res_map:
                        peer_res_map[serv] = copy.deepcopy(j.res_map[serv])
                    #peer_res_map = j.res_map
                    #peer_res_map = copy.deepcopy(j.res_map)
                    self.logger.debug("Alloc vec = {}".format(j.get_job_alloc_vector))
                    self.deallocate(j.gpus, j, revert_iter=True, time=time)
                    self.logger.debug("After dealloc : {}".format(peer_res_map))

                    # We deallocated it . So get demand vec
                    peer_demand_vec = j.get_job_demand_vector
                    can_adjust, new_demand_vec_peer = self._make_fair_share(j, j.get_job_demand_vector)
                    if can_adjust:
                        peer_demand_vec =  new_demand_vec_peer
                    #peer_res_map = {}
                    for serv in peer_res_map.keys():
                        gpu_share = peer_res_map[serv]['gpu']/len(gpus_realloc)
                        peer_demand_vec_share = [res*gpu_share for res in peer_demand_vec]
                        peer_alloc_map = self._vector_to_map(peer_demand_vec_share)
                        peer_res_map[serv] = peer_alloc_map
                        self.logger.debug("Server {} avail = {}".format(serv.server_id, serv.availability_stats()))

                    self.logger.debug("After adjust : {}".format(peer_res_map))
                    self._do_allocate(
                      gpus_realloc,  j, time, res_map=peer_res_map)
                    self.logger.debug("   After adj : Server {} : {} ".format(serv.server_id, serv.availability_stats()))
            self.logger.debug("Job :{} demand vec = {}".format(str(job), demand_vec))
            return self._tune(job, demand_vec, job_gpu_deficit, False, False, True, _call_allocate, available_gpus, time, fair)

        else:
            raise Exception("Cannot adjust job")
        
    def _get_underutilized_servers(self, num_gpus, available_gpus, consolidate=False):
        self.logger.debug("Num gpus:{}, avail_gpus:{}".format(num_gpus, len(available_gpus)))
        if num_gpus > 1:
        #if consolidate:
            for serv in self.servers:
                if list(serv.availability_stats())[0] >= num_gpus:
                    return {serv:num_gpus}
        # DO the heap here
        server_handle_to_available_gpus = {}
        for gpu in available_gpus:
            if gpu.server_handle not in server_handle_to_available_gpus:
                server_handle_to_available_gpus[gpu.server_handle] = []
            server_handle_to_available_gpus[gpu.server_handle].append(gpu)

        pq = [(len(gpus), m) for m, gpus in server_handle_to_available_gpus.items()]
        server_map = {}
        heapq.heapify(pq)
        while num_gpus > 0 and len(pq) > 0: 
            gpus, serv = heapq.heappop(pq)
            if gpus >= num_gpus:
                server_map[serv] = num_gpus
                return server_map
            else:
                server_map[serv] = gpus
                num_gpus -= gpus
        return server_map

    def _reallocate_peer(self, demand_vec, avail_vec, server_handle):
        self.logger.debug("Reallocation in Server {} : {} ".format(server_handle.server_id, server_handle.availability_stats()))
        spare_res_need = [max(0, x1 - x2) for (x1, x2) in zip(demand_vec, avail_vec)]
        if all(v == 0 for v in spare_res_need):
            self.logger.debug("No job to change, spare:{}".format(spare_res_need))
            return []

        jobs_to_realloc = []
        job_list = self.server_job_schedule[server_handle.server_id]
        job_list.sort(key=lambda x: (len(x.server_ids()), -((x.get_job_alloc_vector)[1]/(x.get_job_alloc_vector)[0]), -((x.get_job_alloc_vector)[2]/(x.get_job_alloc_vector)[0])))

        for j in job_list:
            self.logger.debug("   {}:{}".format(str(j), j.get_job_alloc_vector))
        for j in job_list:
           # Single server jobs
           job_gpus_this_server = len(j.get_gpu_share(server_handle.server_id))
           #job_server_stats = j.res_map[server_handle]
           #job_gpus_this_server = job_server_stats['gpu']
           job_fair = [x*job_gpus_this_server for x in self.per_server_size_fair]
           job_gpu_share = job_gpus_this_server/(j.get_job_alloc_vector)[0]
           #job_gpu_share = job_gpus_this_server/self.per_server_size_fair[0]
           job_alloc_share = [res*job_gpu_share for res in j.get_job_alloc_vector]
           job_excess_vec = [max(0, x1 - x2) for (x1, x2) in zip(job_alloc_share, job_fair)]
           diff = [max(0, x2 - x1) for (x1, x2) in zip(job_excess_vec, spare_res_need)]
           self.logger.debug("Diff vec = {} for {}, job_alloc_share={}".format(diff, str(j), job_alloc_share))
           if all(v == 0 for v in diff):
               jobs_to_realloc.append(j)
               return jobs_to_realloc
           elif all(v > 0 for v in diff):
               continue
           else:
               jobs_to_realloc.append(j)
               spare_res_need = diff
        return jobs_to_realloc 
        
    def _make_fair_share(self, job, demand_vec):
        must_switch = False
        new_demand_vec = copy.deepcopy(demand_vec)
        for demand, fair in zip(demand_vec, self.per_server_size_fair):
            if demand > fair*job.job_gpu_demand:
                must_switch = True
        if must_switch:
            new_demand_vec[1] =  self.per_server_size_fair[1] * job.job_gpu_demand
            new_demand_vec[2] =  self.per_server_size_fair[2] * job.job_gpu_demand
            new_demand_vec[3] =  self.per_server_size_fair[3] * job.job_gpu_demand
            #reset speedup to default
            job.synergy_speedup = 1
            return True, new_demand_vec
        return False, None


    def _vector_to_map(self, job_demand_vector):
        alloc_map = {}
        alloc_map["gpu"] = job_demand_vector[0]
        alloc_map["cpu"] = job_demand_vector[1]
        alloc_map["mem"] = job_demand_vector[2]
        alloc_map["sspeed"] = job_demand_vector[3]
        return alloc_map


    """ 
    Deallocate gpus from job

    If res_map is None, cpu, and mem proportional
    to the #gpus assigned to the job is realeased.
    If job <4 GPU, 16 CPU, 200GB mem), and deaddloc 2 GPU,
    8 CPU, 100GB mem is released.
    """
    def deallocate(self, gpus, job, revert_iter=False, time=0):
        # Map of items to be deallocated corresponding to the GPU
        release_map = job.res_map
        #release_map = {}
        #release_map = self.get_allocation_map(gpus)

        #res_map = job.res_map
        try:
            for _server in release_map.keys():
                _server.deallocate(release_map[_server])
        except:
            raise("Cannot deallocate server resources")

        job.deallocate(gpus, release_map=release_map, revert_iter=revert_iter, time=time, simulate=self.simulate)

        # print gpu deallocation and free gpu 
        for gpu in gpus:
            self.logger.debug("%s - %s", gpu, job)
            gpu.free()


    def print_cluster_info(self):
        for gpu in self.gpus:
            print("{}".format(gpu))

    def __getstate__(self):
        d = self.__dict__.copy()
        if 'logger' in d:
            d['logger'] = d['logger'].name
        return d

    def __setstate__(self, d):
        if 'logger' in d:
            d['logger'] = logging.getLogger(d['logger'])
        self.__dict__.update(d)
