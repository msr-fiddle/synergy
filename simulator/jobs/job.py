import logging
import math
import collections
import copy

def nested_add(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, collections.Mapping):
            tmp = nested_add(orig_dict.get(key, { }), val)
            orig_dict[key] = tmp
        elif isinstance(val, list):
            orig_dict[key] = (orig_dict.get(key, []) + val)
        else:
            if key not in orig_dict:
                orig_dict[key] = new_dict[key]
            else:
                orig_dict[key] += new_dict[key]
    return orig_dict

class Job:

    def __init__(
        self,
        job_id,
        job_arrival_time,
        job_iteration_time,
        job_total_iteration,
        job_gpu_demand,
        job_packing_penalty,
        job_placement_penalty,
        synergy_res_matrix,
        synergy_storage_matrix, 
        tenant_id,
        job_cpu_demand=-1,
        job_mem_demand=-1,
        job_sspeed_demand=-1,
        job_queueing_delay=0,
        cluster_id=0,
        job_priority=0,
        iter_is_duration=False):
        # logger handle
        self.logger = logging.getLogger(__name__)

        # job details
        self.job_id = job_id
        self.job_arrival_time = job_arrival_time
        self.job_iteration_time = job_iteration_time
        self.job_iteration_time_orig = job_iteration_time
        self.job_total_iteration = job_total_iteration
        self.job_duration = job_iteration_time * job_total_iteration
        self.job_gpu_demand = job_gpu_demand
        self.job_cpu_demand = job_cpu_demand
        self.job_mem_demand = job_mem_demand
        self.job_sspeed_demand = job_sspeed_demand

        self.job_gpu_demand_orig = job_gpu_demand
        self.job_cpu_demand_orig = job_cpu_demand
        self.job_mem_demand_orig = job_mem_demand
        self.job_sspeed_demand_orig = job_sspeed_demand

        self.job_packing_penalty = job_packing_penalty
        self.job_placement_penalty = job_placement_penalty
        self.tenant_id = tenant_id
        self.job_queueing_delay = job_queueing_delay
        self.job_priority = job_priority
        self.iter_is_duration = iter_is_duration
        self.job_class_id = -1
        self.job_task = None
        self.job_model = None #Model object
        self.synergy_iter_updated=False
        self.last_round_progress = 0
        self.last_round_attained_time = 0
        self.current_round_iters = 0
        self.current_round_time = 0
        self.synergy_speedup = 1
        self.tput = None
        self.dominant_share = 0

        # job state
        self.gpus = list()
        self.cpu_ids = dict()
        self.num_allocated_gpus = 0
        self.cpus = 0
        self.mem = 0
        self.sspeed = 0
        self.res_map = {}
        self.util_map = {}
        self.demand_map = {}
        self.gpus_last_round = list() 
        self.job_executed_iteration = 0
        self.job_last_execution_time = -1
        self.attained_service_time = 0
        self.lease_extended = False
        self.job_command = None

        self.cpu_val = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:9, 7:12, 8:24}
        self.mem_val = {0:20.83, 1:62.5, 2:125, 3:187.5, 4:250}

    def get_idx(self, id_map, value):
        for k,v in id_map.items():
            if value == v:
                return k
        return None

    def clear_alloc_status(self, simulate=True):
        if not simulate:
            self.gpus = list()
            self.cpu_ids = dict()
            self.num_allocated_gpus = 0
            self.cpus = 0
            self.mem = 0
            self.sspeed = 0
            self.res_map = {}
            self.util_map = {}
            self.demand_map = {}
        return

    # Returns a deepcopy of the self job with the alloc status of the job 
    def copy_with_alloc_status(self, job, simulate=True):
        new_job =  copy.deepcopy(self)
        new_job.gpus = job.gpus
        new_job.cpu_ids = job.cpu_ids
        new_job.num_allocated_gpus = job.num_allocated_gpus
        new_job.cpus = job.cpus
        new_job.mem = job.mem
        new_job.sspeed = job.sspeed
        new_job.res_map = job.res_map
        new_job.util_map = job.util_map 
        new_job.demand_map = job.demand_map
        return new_job
       

    def allocate(self, gpus, time=None, res_map=None, fair=True, simulate=True):
        # check if gpus is a list of objects or singleton object
        try:
            iter(gpus)
            self.gpus.extend(gpus)
        except TypeError:
            self.gpus.append(gpus)

        self.num_allocated_gpus = len(self.gpus)

        if not bool(self.res_map) and res_map is not None:
            # Empty original res_map, but valid res_map
            self.res_map = res_map
        elif res_map is not None:
            self.res_map = nested_add(self.res_map, res_map)
        else:
            # Empry res_map passed in
            raise ValueError("Empty resource allocation map for job {}".format(self.job_id))

        self.update_res()           
        #self.update_utilization()

        # print gpu allocations
        machine_ids = set()
        for gpu in self.gpus:
            self.logger.debug("%s - %s", gpu, self)
            machine_ids.add(gpu.machine_id)

        if not simulate:
            return

        # If spread across machines, apply placement penalty ( eg 0.8 => tput drops to 0.8x )
        job_iteration_time = self.job_iteration_time
        if len(machine_ids) > 1:
            job_iteration_time = job_iteration_time / self.job_placement_penalty

        # Find speedup given res_map

        if not fair:
            job_iteration_time = job_iteration_time / self.synergy_speedup
        
        # update remaining iterations for job
        # time is None in case job is given an allocation to completion
        if time is not None:
            self.last_round_attained_time = self.attained_service_time
            self.attained_service_time += time
            progress_in_iteration = math.floor(time/job_iteration_time)
            self.last_round_progress = self.job_executed_iteration
            self.job_executed_iteration += progress_in_iteration
            if self.job_executed_iteration > self.job_total_iteration:
                # print(self.job_executed_iteration)
                self.job_executed_iteration = self.job_total_iteration
            self.logger.debug("Alloc iter for {} = {}. Now {}".format(str(self), progress_in_iteration, self.job_executed_iteration))
        else:
            self.attained_service_time += self.job_total_iteration * job_iteration_time
            self.job_executed_iteration = self.job_total_iteration


    # Check if all GPU iterators have returned in multi-GPU deployment
    def ready_to_deallocate(self):
        self.num_allocated_gpus -= 1
        if self.num_allocated_gpus <= 0:
            self.num_allocated_gpus = 0
            return True
        return False


    # Map of deallocations to be performed
    def deallocate(self, gpus, release_map=None, revert_iter=False, time=0, simulate=True):

        #dealloc to adjust jobs
        if revert_iter:
            #rollback one round duration
            self.attained_service_time -= time
            #progress_in_iteration = math.floor(time/self.job_iteration_time)
            #if self.job_executed_iteration == self.job_total_iteration:
            self.job_executed_iteration = self.last_round_progress
            #self.job_executed_iteration -= progress_in_iteration
            #self.logger.debug("Rollback for {} = {}".format(str(self), progress_in_iteration))
            #if self.job_executed_iteration <= 0:
            #    self.job_executed_iteration = 0
            if self.attained_service_time <= 0:
                self.attained_service_time = 0
            
            
        self.logger.debug("Iters completed for {} = {}/{}".format(str(self), self.job_executed_iteration, self.job_total_iteration))
        # print gpu deallocations

        #fraction = len(gpus) / len(self.gpus)

        for gpu in gpus:
            self.logger.debug("%s - %s", gpu, self)
       
        self.logger.debug("Dealloc job G:{}, C:{}".format(self.gpus, self.cpu_ids)) 
        # non-deallocated gpus stay with the job
        self.gpus = [gpu for gpu in self.gpus if gpu not in gpus]
        self.num_allocated_gpus = len(self.gpus)

        # Assume all GPUs are dealloc at once in deployment
        if not simulate and not revert_iter:
            for server in self.res_map:
                cpus = self.cpu_ids[server.server_id]   
                server.add_cpus_available(cpus) 
                del self.cpu_ids[server.server_id]

        # update gpus allocated in last round
        self.gpus_last_round = gpus

        # deallocate other resources prop to gpus
        if release_map is None:
            release_map = self.res_map

        for _server in self.res_map:
            res = self.res_map[_server]
            util = self.util_map[_server]
            demand = self.demand_map[_server]
            # proportional to gpus being deallocated in the server
            alloc_share = release_map[_server]['gpu']/self.res_map[_server]['gpu']     
#            print("Job :{}, alloc_share:{}, Server:{}".format(str(self), alloc_share, _server.server_id))
            if 'cpu' in res:
                res['cpu'] -= math.floor(res['cpu']*alloc_share)
                _server.cpu_true_utilization -= math.floor(util['cpu']*alloc_share)
                _server.cpu_demand -= math.floor(demand['cpu']*alloc_share)
            if 'gpu' in res:
                res['gpu'] -= math.floor(res['gpu']*alloc_share)
            if 'mem' in res:
                res['mem'] -= res['mem']*alloc_share
                _server.mem_true_utilization -= util['mem']*alloc_share
                _server.mem_demand -= demand['mem']*alloc_share
            if res['gpu'] == 0:
                res['sspeed'] = 0
            #if 'sspeed' in res:
                _server.sspeed_true_utilization -= util['sspeed']*alloc_share
                _server.sspeed_demand -= demand['sspeed']*alloc_share
#            print("Dealloc demand : ", _server.cpu_demand, _server.mem_demand, _server.sspeed_demand, self.job_cpu_demand)
#            print("Dealloc util : ", _server.cpu_true_utilization, _server.mem_true_utilization, _server.sspeed_true_utilization)
            assert(_server.cpu_demand >= 0)
            assert(_server.mem_demand >= 0)
            assert(_server.sspeed_demand >= 0)
        self.remove_unused_servers()
        self.update_res()           

    def update_res(self):
        # Update allocations for the job based on res_map
        cpus = mem = sspeed = 0
        for _server in self.res_map:
            res = self.res_map[_server]
            if 'cpu' in res:
                cpus += res['cpu']
            if 'mem' in res:
                mem += res['mem']
            if 'sspeed' in res:
                sspeed += res['sspeed']
        # aggregate utilization for this job across servers on whcih it might run
        self.cpus = cpus
        self.mem = mem
        self.sspeed = sspeed

    def update_utilization(self):
        # If more than required CPU, mem or sspped are allocated to the
        # job, appropriately update the server resource utilization
        # This approximates the server resource utilization one 
        # would get with tools like dstat
        #self.util_map = copy.deepcopy(self.res_map)

        # number of servers acroiss which this job runs
        # cpu_deficit is the total underutilized cpu per job,
        # so equally distribute it across servers on which job runs
        num_servers = len(self.res_map.keys()) 

        for _server in self.res_map:
            self.util_map[_server] = copy.deepcopy(self.res_map[_server])
            self.demand_map[_server] = copy.deepcopy(self.res_map[_server])

            demand = self.demand_map[_server]       
            util = self.util_map[_server]

            #print(demand)
            #print(util)
            #if num_servers > 1:
            #    print("Before : ", _server.cpu_demand, _server.mem_demand, _server.sspeed_demand, self.job_cpu_demand, num_servers, self.job_gpu_demand)
            # GPUs for this job allocated in the particular server
            alloc_share = self.res_map[_server]['gpu']/len(self.gpus)      
 
            demand['cpu'] += math.floor(self.get_cpu_deficit*alloc_share)
            _server.cpu_demand += math.floor(self.get_cpu_deficit*alloc_share)
            demand['mem'] += self.get_mem_deficit*alloc_share
            _server.mem_demand += self.get_mem_deficit*alloc_share
            demand['sspeed'] += self.get_sspeed_deficit*alloc_share
            _server.sspeed_demand += self.get_sspeed_deficit*alloc_share
            #if num_servers > 1:
            #    print("After : ", _server.cpu_demand, _server.mem_demand, _server.sspeed_demand, self.job_cpu_demand, num_servers)
            if _server.cpu_demand < 0:
                print(self.get_cpu_deficit, num_servers, math.floor(self.get_cpu_deficit*alloc_share))
                print(self.res_map)
            assert(_server.cpu_demand >= 0)
            assert(_server.mem_demand >= 0)
            assert(_server.sspeed_demand >= 0)

            if self.get_cpu_deficit < 0:
               # Allocated cpus are underutilized
               # Add because deficit is -ve
               util['cpu'] += math.floor(self.get_cpu_deficit*alloc_share)
               _server.cpu_true_utilization += math.floor(self.get_cpu_deficit*alloc_share)

            if self.get_mem_deficit < 0:
               # Allocated mem is underutilized
               # Add because deficit is -ve
               util['mem'] += self.get_mem_deficit*alloc_share
               _server.mem_true_utilization += self.get_mem_deficit*alloc_share

            if self.get_sspeed_deficit < 0:
               # Allocated sspeed are underutilized
               # Add because deficit is -ve
               util['sspeed'] += self.get_sspeed_deficit*alloc_share
               _server.sspeed_true_utilization += self.get_sspeed_deficit*alloc_share
            assert(_server.cpu_true_utilization >= 0)
            assert(_server.mem_true_utilization >= 0)
            assert(_server.sspeed_true_utilization >= 0)
  
 
    def remove_unused_servers(self):
        serv_to_del = []
        for _server in self.res_map:
            if self.res_map[_server]['gpu'] <= 0:
                serv_to_del.append(_server)
        for _server in serv_to_del:
            del self.res_map[_server]

    def get_time_since_last_execution(self, time):
        if self.job_last_execution_time == -1:
            return (time - self.job_arrival_time)
        # time elapsed since last execution
        return (time - self.job_last_execution_time)
    
    def get_gpu_deficit(self):
        # job unmet demand in number of gpus
        return (self.job_gpu_demand - len(self.gpus))
 
    @property
    def get_job_demand_vector(self):
        return [self.get_gpu_deficit(), self.get_cpu_deficit, self.get_mem_deficit, self.get_sspeed_deficit, 0]

    @property
    def get_job_alloc_vector(self):
        return [len(self.gpus), self.cpus, self.mem, self.sspeed, 0]
  
    def get_gpu_share(self, server_id):
        gpus = []
        for gpu in self.gpus:
            if gpu.machine_id == server_id:
                gpus.append(gpu)
        return gpus 
 
    @property 
    def get_cpu_deficit(self):
        # job unmet demand in number of cpus
        return (self.job_cpu_demand - self.cpus)

    @property
    def get_mem_deficit(self):
        # job unmet demand in memory
        return (self.job_mem_demand - self.mem)

    @property 
    def get_sspeed_deficit(self):
        # job unmet demand in storage speed
        return (self.job_sspeed_demand - self.sspeed)

    def get_remaining_weighted_duration(self, size_vec, fair=True):
        remaining_iteration = self.job_total_iteration -\
            self.job_executed_iteration 
        dur = 0
        if fair:
            dur =  (self.job_iteration_time * remaining_iteration)
        else:
            dur =  (self.job_iteration_time * remaining_iteration / self.synergy_speedup)

        d_gpu, d_cpu, d_mem, _, _ = self.get_job_demand_vector
        _,_,gpus, cpus, mem,_,_ = size_vec
        norm_demand = [ d_gpu/gpus, d_cpu/cpus, d_mem/mem]
        weighted_demand = [item*dur for item in norm_demand]
        #self.logger.info("Share for {} = {} : {} : {:.2f}val : {:.2f}s, {}speedup, {}iter, {:.2f}s".format(str(self), norm_demand, weighted_demand, sum(weighted_demand), dur, self.synergy_speedup, remaining_iteration, self.job_iteration_time)) 
        return sum(weighted_demand)

    def remaining_duration(self, fair=True):
        # job remaining best-scenario duration
        remaining_iteration = self.job_total_iteration -\
            self.job_executed_iteration 
        if fair:
            return (self.job_iteration_time * remaining_iteration)
        else:
            return (self.job_iteration_time * remaining_iteration / self.synergy_speedup)
        #return (self.job_iteration_time * remaining_iteration)

    def ideal_duration(self, fair=True):
        if fair:
            return (self.job_iteration_time * self.job_total_iteration)
        else:
            return (self.job_iteration_time * self.job_total_iteration / self.synergy_speedup)


    def remaining_service(self):
        # job remaining best-case gpu-time
        remaining_iteration = self.job_total_iteration -\
            self.job_executed_iteration 
        return (self.job_iteration_time * remaining_iteration *\
                self.job_gpu_demand) 

    # In synergy adjust, diff iters might have run with diff 
    # dur, so track elapsed time interms of simulator round dur
    def get_attained_service_time(self):
        return self.attained_service_time*self.job_gpu_demand

    # size_vec: (racks, servers, gpus, cpus, mem, sspeed, net)
    def get_dominant_share(self, size_vec):
        _,_,gpus, cpus, mem,_,_ = size_vec
        #self.logger.info("Max avail for {} = {},{},{}".format(str(self), gpus, cpus, mem)) 
        gpu_share = self.job_gpu_demand/gpus    
        cpu_share = self.job_cpu_demand/cpus    
        mem_share = self.job_mem_demand/mem
        self.dominant_share = max(gpu_share, cpu_share, mem_share) 
        #self.logger.info("Share for {} = {:.4f},{:.4f},{:.4f}, max={:.4f}".format(str(self), gpu_share, cpu_share, mem_share, self.dominant_share)) 
        
        return self.dominant_share   


    def attained_service(self):
        # job allocated gpu-time
        return (self.job_iteration_time * self.job_executed_iteration *\
                self.job_gpu_demand)

    def finish_time_fair_metric(self, time, fair=True):
        #Approx estimate
        return ((time - self.job_arrival_time + self.remaining_duration(fair)) / self.ideal_duration(fair=True)) 


    def is_finished(self):
        # job finish criteria
        if self.job_executed_iteration >= self.job_total_iteration:
            #self.logger.info("Job Finished - {}/{}".format(self.job_executed_iteration,self.job_total_iteration))
            return True
        return False

    def prefers_consolidation(self):
        if self.job_gpu_demand > 1:
            return True
        return False

    def server_ids(self):
        server_ids = set()
        for gpu in self.gpus:
            server_ids.add(gpu.machine_id)
        return list(server_ids)


    def __eq__(self, other):
        return (self.job_id == other.job_id and\
                self.tenant_id == other.tenant_id)

    def __str__(self):
        return "job:%s:%s:%s:%s:%s (%s s,%s)" % (self.job_id, self.job_model.model_name, 
             self.job_gpu_demand, self.job_cpu_demand, self.job_mem_demand, self.job_arrival_time, self.job_total_iteration)
    
    def __getstate__(self):
        d = self.__dict__.copy()
        if 'logger' in d:
            d['logger'] = d['logger'].name
        return d

    def __setstate__(self, d):
        if 'logger' in d:
            d['logger'] = logging.getLogger(d['logger'])
        self.__dict__.update(d)

