import logging
import numa
import bisect
import numpy as np
from resources.gpu import GPU
from resources.server_config import DefaultServerConfig
'''
Creates a server with the specified number of GPUs.
If unspecified, creates a server with 8 homogenous GPUs.
Assumes that GPUs in a server are always homogenous.
If CPU, DRAM, and disk_bw not specificed, uses the default values
of those objects.
'''

class Server():
    
    def __init__(self,
                 rack_id=None,
                 server_id=None,
                 server_config=None,
                 start_gpu=None,
                 ip=None,
                 port=None,
                 start_gpu_deploy=0,
                 start_cpu_deploy=0,
                 numa_aware=False):
        self.logger = logging.getLogger(__name__)
        self.server_id = 0
        self.rack_id = 0
        self.start_cpu_deploy = int(start_cpu_deploy)
        self.start_gpu_deploy = int(start_gpu_deploy)
        self.numa_aware = int(numa_aware)
       

        if server_config is not None:
            self.server_config = server_config
        else:
            self.server_config = DefaultServerConfig()
           

        if server_id is not None:
            self.server_id = server_id

        if rack_id is not None:
            self.rack_id = rack_id

        # If nopt simulation, the IP, port and rpc client of the worker
        self.ip = ip
        if port is not None:
            self.port = int(port)
        else:
            self.port = port
        self.rpc_client = None


        # Max resource availability in the server
        self.num_cpus = self.server_config.cpu
        self.memory = self.server_config.mem
        self.storage_speed = self.server_config.sspeed
        self.num_gpus = self.server_config.gpu
        self.network = self.server_config.net

        self.cpus_available = list(range(self.start_cpu_deploy + 0, self.start_cpu_deploy + self.num_cpus))
        self.cpu_map_numa = dict()
        self.cpu_map_numa_ref = dict()
        if self.numa_aware:
            self.init_numa()


        # Resources allocated in the server
        self.gpu_used = 0
        self.cpu_used = 0
        self.mem_used = 0
        self.sspeed_used = 0
        self.net_used = 0

        # Resources on hold during recursive allocation
        self.gpu_on_hold = 0 
        self.cpu_on_hold = 0 
        self.mem_on_hold = 0 
        self.sspeed_on_hold = 0 
        self.net_on_hold = 0 

        # Resources actually utilized by the jobs its allocated to
        # This is updated by the running jobs
        self.cpu_true_utilization = 0
        self.mem_true_utilization = 0
        self.sspeed_true_utilization = 0
        self.net_true_utilization = 0


        # Resources demanded by the jobs 
        # This is updated by the running jobs
        self.cpu_demand = 0
        self.mem_demand = 0
        self.sspeed_demand = 0
        self.net_demand = 0


        self.gpus = []
        self.allocated_gpus = []
        self.gpus = self.get_gpus(start_gpu)

    def init_numa(self):
        # Populate self.cpu_map_numa - one list entry for each numa node
        num_numa_nodes = numa.get_max_node() + 1
        for i in range(0, num_numa_nodes):
            self.cpu_map_numa[i] = list(numa.node_to_cpus(i))
            self.cpu_map_numa_ref[i] = list(numa.node_to_cpus(i))

        self.logger.info("Numa nodes = {}".format(num_numa_nodes))
        self.logger.info("Numa node CPUs = {}".format(self.cpu_map_numa))


    def get_cpus(self, num_cpus):
        if not self.numa_aware:
            if len(self.cpus_available)  < num_cpus:
                return None
            cpus = self.cpus_available[:num_cpus]
            del self.cpus_available[:num_cpus]
            self.logger.info("Get {} CPUs : {}".format(num_cpus, cpus))
            return cpus
        else:
            # FInd the numa node with fewest available CPUs
            # that can satisy the current demands
            candidate_nodes = [n for n, cpus in self.cpu_map_numa.items()\
                          if len(cpus) >= num_cpus]
            if len(candidate_nodes) > 0:
                target_numa = sorted(candidate_nodes, key = lambda n : len(self.cpu_map_numa[n]))[0]
                cpus = self.cpu_map_numa[target_numa][:num_cpus]
                del self.cpu_map_numa[target_numa][:num_cpus]
                self.logger.info("Get {} CPUs : {}".format(num_cpus, cpus))
                return cpus

            sorted_numa_list = sorted(self.cpu_map_numa.keys(), key = lambda n : len(self.cpu_map_numa[n]))
            cpus = []
            i = 0
            while len(cpus) < num_cpus:
                cpus_needed = num_cpus - len(cpus)
                numa_id = sorted_numa_list[i]
                if cpus_needed < len(self.cpu_map_numa[numa_id]): 
                    cpu_ids = self.cpu_map_numa[numa_id][:cpus_needed]
                    del self.cpu_map_numa[numa_id][:cpus_needed]
                    cpus.extend(cpu_ids)
                else:
                    cpu_ids = self.cpu_map_numa[numa_id]
                    self.cpu_map_numa[numa_id] = []
                    cpus.extend(cpu_ids)
                i += 1
            return cpus
 
                

    def remove_cpus_available(self, cpu_ids):
        if self.numa_aware:
            for cpu in cpu_ids:
                for numa_id, cpu_list in self.cpu_map_numa.items():
                    if cpu in cpu_list:
                        cpu_list.remove(cpu)
                        self.cpu_map_numa[numa_id] = cpu_list
                    
        else:
            for cpu in cpu_ids:
                self.cpus_available.remove(cpu)


    def add_cpus_available(self, cpu_ids):
        if self.numa_aware:
            for cpu in cpu_ids:
                for numa_id, cpu_list in self.cpu_map_numa_ref.items():
                    if cpu in cpu_list:
                        bisect.insort(self.cpu_map_numa[numa_id], cpu)
           
        else:
            for cpu in cpu_ids:
                bisect.insort(self.cpus_available, cpu)


    def print_cpus_available(self):
        if self.numa_aware:
            return self.cpu_map_numa
        else:
            return self.cpus_available
        

    def get_gpus(self, start_gpu):
        if start_gpu is None:
            start_gpu = 0
        job_id  = -1
        tenant_id = -1

        gpus = []

        for gpu_id in np.arange(start_gpu, start_gpu+self.num_gpus):
            gpu = GPU(self.rack_id, self.server_id, gpu_id, job_id, 
                     tenant_id, server_handle=self)
            gpus.append(gpu)
        return gpus



    # Update allocation of cpu, gpu, mem in the server
    def allocate(self, res_map):
        #print(res_map)
        #print(type(res_map))
        if not isinstance(res_map, dict):
            raise ValueError("Resource map is invalid")

        # No strict allocation restriction on storage speed
        for res in res_map:
            if 'cpu' in res:
                assert(res_map[res] <= self.cpu_available)
                self.cpu_used += res_map[res]
                self.cpu_true_utilization += res_map[res]
                self.cpu_demand += res_map[res]
                assert(int(self.cpu_true_utilization) <= int(self.cpu_used))
                assert(int(self.cpu_true_utilization) >= 0)
            if 'mem' in res:
                assert(res_map[res] <= self.mem_available)
                self.mem_used += res_map[res]
                self.mem_true_utilization += res_map[res]
                self.mem_demand += res_map[res]
                assert(int(self.mem_true_utilization) <= int(self.mem_used))
                assert(int(self.mem_true_utilization) >= 0)
            if 'gpu' in res:
                assert(res_map[res] <= self.gpu_available)
                self.gpu_used += res_map[res]
            if 'sspeed' in res:
                self.sspeed_used += res_map[res]
                self.sspeed_true_utilization += res_map[res]
                self.sspeed_demand += res_map[res]
                assert(int(self.sspeed_true_utilization) <= int(self.sspeed_used))
                assert(int(self.sspeed_true_utilization) >= 0)
        self.logger.debug("Allocate [{}] : gpus:{}, cpus:{}, mem:{}, sspeed:{}".format(self.id, self.gpu_available, self.cpu_available, self.mem_available, self.sspeed_available))

    def deallocate(self, res_map):
        if not isinstance(res_map, dict):
            raise ValueError("Resource map is invalid")
        #print(res_map)
        for res in res_map:
            if 'cpu' in res:
                assert(self.cpu_used >= res_map[res])
                self.cpu_used -= res_map[res]
            if 'mem' in res:
                assert(res_map[res] <= self.mem_used)
                self.mem_used -= res_map[res]
            if 'gpu' in res:
                assert(res_map[res] <= self.gpu_used)
                self.gpu_used -= res_map[res]
            if 'sspeed' in res:
                self.sspeed_used -= res_map[res]
        self.logger.debug("Deallocate [{}] : gpus:{}, cpus:{}, mem:{}, sspeed:{}".format(self.id, self.gpu_available, self.cpu_available, self.mem_available, self.sspeed_available))
         


    @property
    def id(self):
        return self.server_id

    @property
    def cpu_available(self):
        return self.num_cpus - self.cpu_used

    @property
    def mem_available(self):
        return self.memory - self.mem_used
 
    @property
    def sspeed_available(self):
        return self.storage_speed - self.sspeed_used

    @property
    def gpu_available(self):
        return self.num_gpus - self.gpu_used

    @property
    def net_available(self):
        return self.network - self.net_used

    def alloc_stats(self, percent=False):
        # Gpu, cpu, mem, sspeed, net used
        if percent:
            return (self.gpu_used/self.num_gpus*100, 
                self.cpu_used/self.num_cpus*100, 
                self.mem_used/self.memory*100, 
                self.sspeed_used/self.storage_speed*100, 
                self.net_used/self.network*100)
        else:
            return (self.gpu_used, 
                self.cpu_used, 
                self.mem_used, 
                self.sspeed_used, 
                self.net_used)

    def utilization_stats(self, percent=False):
        # Gpu, cpu, mem, sspeed, net true utilization
        if percent:
            return (self.gpu_used/self.num_gpus*100, 
                self.cpu_true_utilization/self.num_cpus*100, 
                self.mem_true_utilization/self.memory*100, 
                self.sspeed_true_utilization/self.storage_speed*100, 
                self.net_true_utilization/self.network*100)
        else:
            return (self.gpu_used, 
                self.cpu_true_utilization, 
                self.mem_true_utilization, 
                self.sspeed_true_utilization, 
                self.net_true_utilization)

    def demand_stats(self, percent=False):
        # Gpu, cpu, mem, sspeed, net demand at this server
        if percent:
            return (self.gpu_used/self.num_gpus*100, 
                self.cpu_demand/self.num_cpus*100, 
                self.mem_demand/self.memory*100, 
                self.sspeed_demand/self.storage_speed*100, 
                self.net_demand/self.network*100)
        else:
            return (self.gpu_used, 
                self.cpu_demand, 
                self.mem_demand, 
                self.sspeed_demand, 
                self.net_demand)

    def availability_stats(self, percent=False):
        # Gpu, cpu, mem, sspeed, net demand at this server
        if percent:
            return (self.gpu_available/self.num_gpus*100, 
                self.cpu_available/self.num_cpus*100, 
                self.mem_available/self.memory*100, 
                self.sspeed_available/self.storage_speed*100, 
                self.net_available/self.network*100)
        else:
            return (self.gpu_available, 
                self.cpu_available, 
                self.mem_available, 
                self.sspeed_available, 
                self.net_available)


    def availability_stats_with_hold(self, percent=False):
        # Gpu, cpu, mem, sspeed, net demand at this server
        return (self.gpu_available - self.gpu_on_hold, 
                self.cpu_available  - self.cpu_on_hold, 
                self.mem_available - self.mem_on_hold, 
                self.sspeed_available - self.sspeed_on_hold, 
                self.net_available - self.net_on_hold)

    def release_held_resources(self):
        self.gpu_on_hold = 0
        self.cpu_on_hold = 0
        self.mem_on_hold = 0
        self.sspeed_on_hold = 0
        self.net_on_hold = 0

    def hold_resources(self, demand_vec):
        self.gpu_on_hold += demand_vec[0]
        self.cpu_on_hold += demand_vec[1] 
        self.mem_on_hold += demand_vec[2] 
        self.sspeed_on_hold += demand_vec[3] 
        self.net_on_hold += demand_vec[4] 
        
    def __getstate__(self):
        d = self.__dict__.copy()
        if 'logger' in d:
            d['logger'] = d['logger'].name
        return d

    def __setstate__(self, d):
        if 'logger' in d:
            d['logger'] = logging.getLogger(d['logger'])
        self.__dict__.update(d)


    @property
    def gpu_ids(self):
        gpu_ids = []
        for gpu in self.gpus:
            gpu_ids.append(gpu.gpu_id)

        return gpu_ids

    @property
    def gpu_list(self):
        return self.gpus

    @property
    def free_gpu_list(self):
        free = []
        for gpu in self.gpus:
            if gpu not in self.allocated_gpus:
                free.append(gpu)

        return free

    def __cmp__(self,other):
        return cmp(self.server_id, other.server_id)


    def __lt__(self,other):
        return self.server_id < other.server_id

    def print_server_stats(self):
        for gpu in self.gpus:
            print("{}".format(gpu))
