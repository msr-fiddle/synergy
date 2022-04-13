import os
import sys
import cvxpy as cp
import numpy as np
import logging
import argparse
import random
import time
import copy
import gurobipy as gp

from jobs.workload import Workload

GPU_PER_SERVER=8

class Solver():
    def __init__(
        self,
        num_servers=1,
        cpu_per_server=24,
        mem_per_server=500,
        gpu_per_server=GPU_PER_SERVER,
        debug=False):
        
        self.logger = logging.getLogger(__name__)
        self.debug = debug
        self.debug_verbose = False

        self.num_servers = num_servers
        self.cpu_per_server = cpu_per_server
        self.mem_per_server = mem_per_server
        self.gpu_per_server = gpu_per_server
        self.gpus = gpu_per_server * num_servers
        self.cpus = cpu_per_server * num_servers
        self.mem = mem_per_server * num_servers
        self.fair_cpu = self.cpus/self.gpus
        self.fair_mem = self.mem/self.gpus
        self.objective = 0
        self.constraints = []

        self.jobs = None

        # Results
        self.Y = None
        self.X = None
        self.jid_to_idx = {}

    def populate_jid_map(self):
        for idx,job in enumerate(self.jobs):
            self.jid_to_idx[job.job_id] = idx
    
    def clear_stats(self):
        self.objective = 0
        self.constraints = []
        self.jobs = None
        self.Y = None
        self.X = None
        self.jid_to_idx = {}

    def SolveFirstLP(self, jobs, ilp=False, solver=None):
        objective = 0
        self.jobs = jobs
        self.populate_jid_map()
        self.Y = {}
        self.job_to_id = {}

        for job in self.jobs:
            self.Y[job.job_id] = cp.Variable(job.tput.shape, boolean=ilp)
            self.job_to_id[job.job_id] = job

            objective = objective + cp.sum(cp.multiply(job.tput, self.Y[job.job_id]))
        self.objective = cp.Maximize(objective)

        # Populate constrants
        cpu_constraint = 0
        for job in jobs:
             cpu_val = np.array([list(job.cpu_val.values())])
             cpu_val_trans = np.transpose(cpu_val)
             cpu_constraint = cpu_constraint + cp.sum(cp.multiply(cpu_val_trans, self.Y[job.job_id]))
             #print(cpu_val_trans, cpu_val_trans.shape)
        self.constraints.append(cpu_constraint <= self.cpus)


        mem_constraint = 0
        for job in jobs:
             mem_val = np.array([list(job.mem_val.values())])
             mem_constraint = mem_constraint + cp.sum(cp.multiply(mem_val, self.Y[job.job_id]))
        self.constraints.append(mem_constraint <= self.mem)

        config_constraint = 0
        for job in jobs:
             config_constraint = cp.sum(self.Y[job.job_id])
             self.constraints.append(config_constraint == 1)

        fair_share_constraint = 0
        for job in jobs:
             fair_share_constraint = cp.sum(cp.multiply(job.tput, self.Y[job.job_id]))
             fair_cpu_idx = job.get_idx(job.cpu_val, self.fair_cpu*job.gpu_demand)
             fair_mem_idx = job.get_idx(job.mem_val, self.fair_mem*job.gpu_demand)
             fair_share = job.tput[fair_cpu_idx][fair_mem_idx]
             self.constraints.append(fair_share_constraint >= fair_share)

        y_constraint = 0
        for job in jobs:
             y_constraint = self.Y[job.job_id]  
             self.constraints.append( y_constraint >= 0)



        cvxprob = cp.Problem(self.objective, self.constraints)
        if solver is None:
           solver = "GLPK_MI"
        result = cvxprob.solve(solver=solver)
        print(cvxprob.value)
        self.tput = cvxprob.value
        print(cvxprob.status)
        cpu_list = []
        mem_list = []
        #if self.debug:
        for job in jobs:
             print("--"*20)
             print(str(job))
             cpu_val = np.array([list(job.cpu_val.values())])
             cpu_val_trans = np.transpose(cpu_val)
             mem_val = np.array([list(job.mem_val.values())])
             mem = cp.sum(cp.multiply(mem_val, self.Y[job.job_id]))
             cpu = cp.sum(cp.multiply(cpu_val_trans, self.Y[job.job_id]))
             cpu_list.append(cpu.value.round(2))
             mem_list.append(mem.value.round(2))
             fair_share_constraint = cp.sum(cp.multiply(job.tput, self.Y[job.job_id]))
             fair_cpu_idx = job.get_idx(job.cpu_val, self.fair_cpu*job.gpu_demand)
             fair_mem_idx = job.get_idx(job.mem_val, self.fair_mem*job.gpu_demand)
             if fair_cpu_idx is None or fair_mem_idx is None:
                 raise ValueError("Invalid idx in tput map")

             fair_share = job.tput[fair_cpu_idx][fair_mem_idx]
             print("\tEstimated tput={}, Fair share tput ={}".format(fair_share_constraint.value.round(2), fair_share.round(2)))
             print("\tcpu= {}, mem = {}, fair_mem={}".format(cpu.value, mem.value, fair_mem_idx))
             #print("\tcpu= {}, mem = {}, fair_mem={}".format(cpu.value.round(5), mem.value.round(5), fair_mem_idx))

             job.cpu_demand = cpu.value
             #job.cpu_demand = cpu.value.round(5)
             job.mem_demand = mem.value
             job.res_map = {}
             #job.mem_demand = mem.value.round(5)

        print("Total CPU={}, mem= {}".format(sum(cpu_list), sum(mem_list)))

        if self.debug_verbose:
            print("--"*20)
            for job in jobs:
                print("--"*20)
                print(str(job))
                print(self.Y[job.job_id].value.round(2))
                print(job.tput)
                cpu_val = np.array([list(job.cpu_val.values())])
                cpu_val_trans = np.transpose(cpu_val)
                mem_val = np.array([list(job.mem_val.values())])
                mem = cp.sum(cp.multiply(mem_val, self.Y[job.job_id]))
                cpu = cp.sum(cp.multiply(cpu_val_trans, self.Y[job.job_id]))
                cpu_list.append(cpu.value.round(2))
                mem_list.append(mem.value.round(2))
                fair_share_constraint = cp.sum(cp.multiply(job.tput, self.Y[job.job_id]))
                fair_cpu_idx = job.get_idx(job.cpu_val, self.fair_cpu*job.gpu_demand)
                fair_mem_idx = job.get_idx(job.mem_val, self.fair_mem*job.gpu_demand)
                fair_share = job.tput[fair_cpu_idx][fair_mem_idx]
                print("\tEstimated tput={}, Fair share tput ={}".format(fair_share_constraint.value.round(2), fair_share.round(2)))
                print("\tcpu= {}, mem = {}".format(cpu.value.round(2), mem.value.round(2)))
            print("Total CPU={}, mem= {}".format(sum(cpu_list), sum(mem_list)))


    def SolveSecondLP(self, ilp=False, solver='OSQP'):
        objective = 0
        # (i,j) => i= #servers, j=#jobs
        print("Servers={}, jobs={}".format(self.num_servers, len(self.jobs)))
        self.X = cp.Variable((self.num_servers, len(self.jobs)), boolean=ilp)
        constraints = []
    
        gpu_demand_vec = []
        cpu_demand_vec = []
        mem_demand_vec = []
        for job in self.jobs:
            gpu_demand_vec.append(job.gpu_demand)
            cpu_demand_vec.append(job.cpu_demand)
            mem_demand_vec.append(job.mem_demand)

        gpu_demand_np = np.array([gpu_demand_vec])
        cpu_demand_np = np.array([cpu_demand_vec])
        mem_demand_np = np.array([mem_demand_vec])
        print(gpu_demand_np, cpu_demand_np, mem_demand_np)

        #for i in range(self.num_servers):
        constraints.append(cp.sum(cp.multiply(gpu_demand_np, self.X), axis=1) <= self.gpu_per_server)

        constraints.append(cp.sum(cp.multiply(cpu_demand_np, self.X), axis=1) <= self.cpu_per_server)
       
 
        constraints.append(cp.sum(cp.multiply(mem_demand_np, self.X), axis=1) <= self.mem_per_server)

        constraints.append(cp.sum(self.X, axis=0) == 1)
        constraints.append(self.X >= 0)

        cvxprob = cp.Problem(cp.Minimize(1), constraints) 
        if solver is not None:
            result = cvxprob.solve(solver=solver)
        else:
            result = cvxprob.solve()
        print(cvxprob.status)
        print(self.X.value.round(2))
        print("GPU vec = {}".format(cp.sum(cp.multiply(gpu_demand_np, self.X), axis=1).value))
        print("CPU vec = {}".format(cp.sum(cp.multiply(cpu_demand_np, self.X), axis=1).value))
        print("Mem vec = {}".format(cp.sum(cp.multiply(mem_demand_np, self.X), axis=1).value))
        print("-"*45)
        # X is an allocation matrix of serv x jobs
        for job in self.jobs:
            job_idx = self.jid_to_idx[job.job_id]     
            alloc = self.X.value[:,job_idx].round(2)
            for server_id, server_alloc in enumerate(alloc):
                if server_alloc > 0:
                    if server_id not in job.res_map:
                        job.res_map[server_id] = {}

                    job.res_map[server_id]['gpu'] = (server_alloc*job.gpu_demand).round(2)
                    job.res_map[server_id]['cpu'] = (server_alloc*job.cpu_demand).round(2)
                    job.res_map[server_id]['mem'] = (server_alloc*job.mem_demand).round(2)

            print("Job :{}, Res_map={}".format(str(job), job.res_map)) 
        print("-"*45)

    @property
    def is_fractional(self):
        self.frac_X = (self.X.value < 1) == (self.X.value > 0)
        if self.frac_X.any():
            return True
        return False

    def RoundRobin(self):
        print("\n\nFractional allocation. Needs rounding")
        frac_jobs = []
        for job in self.jobs:
            for server_id, alloc in job.res_map.items():
                if not alloc['gpu'].is_integer():
                    frac_jobs.append(job)
                    print("Fractional : {}, {}".format(str(job), job.res_map))
                    break

        current_server_stats = {}
        for server in range(self.num_servers):
            current_server_stats[server] = {}
            current_server_stats[server]['gpu'] = self.gpu_per_server
            current_server_stats[server]['cpu'] = self.cpu_per_server
            current_server_stats[server]['mem'] = self.mem_per_server
        
        for job in filter(lambda job: job not in frac_jobs, self.jobs):
            for server, alloc_map in job.res_map.items():
                current_server_stats[server]['gpu'] -= alloc_map['gpu']
                current_server_stats[server]['cpu'] -= alloc_map['cpu']
                current_server_stats[server]['mem'] -= alloc_map['mem']

        print(current_server_stats)
        unalloc_jobs = []
        for job in frac_jobs:
            job_demand_vec = {}
            job_demand_vec['gpu'] = job.gpu_demand
            #job_demand_vec['cpu'] = self.fair_cpu * job.gpu_demand
            job_demand_vec['cpu'] = min(self.fair_cpu * job.gpu_demand, job.cpu_demand)
            #job_demand_vec['mem'] = self.fair_mem * job.gpu_demand
            job_demand_vec['mem'] = min(self.fair_mem * job.gpu_demand, job.mem_demand)
            print("-"*45)
            print("Alloc job {} : {}".format(str(job), job_demand_vec))
            print("Current avail Map {}".format(current_server_stats)) 
            new_res_map = self.get_server(job_demand_vec, current_server_stats)

            if new_res_map is None:
                unalloc_jobs.append(job)
                print("-"*45)
                print("COULD NOT ALLOCATE {}".format(str(job)))
            else:
                job.res_map = copy.deepcopy(new_res_map)
                current_server_stats = self.nested_reduce(new_res_map, current_server_stats)
                print("Job {}: Res Map {}".format(str(job), new_res_map))

        for job in unalloc_jobs:
            job_demand_vec = {}
            job_demand_vec['gpu'] = job.gpu_demand
            job_demand_vec['cpu'] = job.gpu_demand
            job_demand_vec['mem'] = job.gpu_demand*25
            overlap_server_stats = copy.deepcopy(current_server_stats)
            for serv, avail in overlap_server_stats.items():
                if avail['gpu'] >= 1 and avail['cpu'] < 1:
                    overlap_server_stats[serv]['cpu'] = avail['gpu']
                if avail['gpu'] >= 1 and avail['mem'] < 25:
                    overlap_server_stats[serv]['mem'] = 25*avail['gpu']

            print("-"*45)
            print("Alloc job {} : {}".format(str(job), job_demand_vec))
            print("Current avail Map {}".format(overlap_server_stats)) 
            new_res_map = self.get_server(job_demand_vec, overlap_server_stats)
            if new_res_map is None:
                raise ValueError("Infeasible allocation")
            else:
                job.res_map = copy.deepcopy(new_res_map)
                overlap_server_stats = self.nested_reduce(new_res_map, overlap_server_stats)
                print("Job {}: Res Map {}".format(str(job), new_res_map))

        print("-"*45)
        print("TOTAL FRACTIONAL JOBS = {}, limit={}".format(len(frac_jobs), self.num_servers*3))
        return

    def get_server(self, job_vec, serv_map):
        result_map = {}
        exit = False
        done = False
        round1 = False
        server_map = copy.deepcopy(serv_map)
        new_job_vec = copy.deepcopy(job_vec)
        for server, server_vec in server_map.items():
            if self.fits(new_job_vec,server_vec):
                result_map[server] = new_job_vec 
                print("Fit entirely {}".format(result_map)) 
                return result_map
                        
        print("Second round")
        result_map = {}
        num_rounds = job_vec['gpu']
        new_job_vec = self.get_slice(job_vec, fair=True)
        while job_vec['gpu'] > 0 and not exit:
            print("Loop 1")
            done = False
            for server, server_vec in server_map.items():
                print("{}:{}".format(new_job_vec, server_vec))
                if self.fits(new_job_vec,server_vec):
                    result_map[server] = new_job_vec 
                    job_vec = self.reduce(new_job_vec, job_vec)
                    server_map[server] = self.reduce(new_job_vec, server_map[server])
                    new_job_vec = self.get_slice(job_vec, fair=True)
                    done = True
                    break
            if not done and job_vec['gpu'] > 0:
                exit = True

        if  job_vec['gpu'] == 0:
            return result_map

        print("Third round")
        exit = False
        done = False
        num_rounds = job_vec['gpu']
        new_job_vec = self.get_slice(job_vec, fair=False)
        while job_vec['gpu'] > 0 and not exit:
            print("Loop 2")
            done = False
            for server, server_vec in server_map.items():
                print("{}:{}".format(new_job_vec, server_vec))
                if self.fits(new_job_vec,server_vec):
                    result_map[server] = new_job_vec 
                    job_vec = self.reduce(new_job_vec, job_vec)
                    server_map[server] = self.reduce(new_job_vec, server_map[server])
                    new_job_vec = self.get_slice(job_vec, fair=False)
                    done = True
                    break
            if not done and job_vec['gpu'] > 0:
                exit = True

        if  job_vec['gpu'] == 0:
            return result_map

        # Can't allocate even 1 CPU
        return None

        

    def reduce(self, map1, map2):
        for k,v in map1.items():
            map2[k] -= v
        return map2

    def nested_reduce(self, map1, map2):
        for k,v in map1.items():
            for k1,v1 in v.items():
                map2[k][k1] -= v1
        return map2

    def get_slice(self, in_map, fair=True):
        map1 = copy.deepcopy(in_map)
        if in_map['gpu'] == 0:
            return None
        if fair:
            share = map1['gpu']
            for k, v in map1.items():
                map1[k] = v/share
        else:
            map1['gpu'] = 1
            map1['cpu'] = 1
            map1['mem'] = 25
        return map1

    def fits(self, map1, map2):
        for k,v in map1.items():
            if v > map2[k]:
                return False
        return True


class Job():
    def __init__(
        self,
        name='alexnet',
        iters=0,
        arrival=0,
        gpu_demand=0,
        job_id=0):
     
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.iters = iters
        self.arrival = arrival
        self.gpu_demand = gpu_demand
        self.job_id = job_id
        self.tput = None
        self.cpu_val = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:12}
        self.mem_val = {0:62.5, 1:125, 2:187.5, 3:250}

        # WIll be populated by first LP
        self.cpu_demand = 0
        self.mem_demand = 0

        # Map of server id to a map of "gpu", "cpu", "mem"
        self.res_map = {}

        self.update_tput(self.name)

    def get_idx(self, id_map, value):
        for k, v in id_map.items():
            if value == v:
                return k
        return None

    def __str__(self):
        return "{}:{}:{}:{}".format(self.job_id, self.name, self.iters, self.gpu_demand)


    def update_tput(self, model_name):
        if 'alexnet' in model_name:
             self.tput = np.array([[0.2, 0.2, 0.2, 0.2],
                                   [0.4, 0.4, 0.4, 0.4],
                                   [0.5, 0.5, 0.5, 0.5],
                                   [0.6, 0.6, 0.6, 0.6],
                                   [0.7, 0.7, 0.7, 0.7],
                                   [1.0, 1.0, 1.0, 1.0],
                                   [1.5, 1.5, 1.5, 1.5] ])
        elif 'res18' in model_name:
             self.tput = np.array([[0.2, 0.2, 0.2, 0.2],
                                   [0.4, 0.4, 0.4, 0.4],
                                   [0.5, 0.5, 0.5, 0.5],
                                   [0.55, 0.55, 0.55, 0.55],
                                   [0.6, 0.6, 0.6, 0.6],
                                   [0.9, 0.9, 0.9, 0.9],
                                   [1.0, 1.0, 1.0, 1.0] ])
        elif 'res50' in model_name and self.gpu_demand == 1:
             self.tput = np.array([[0.2, 0.2, 0.2, 0.2],
                                   [0.6, 0.6, 0.6, 0.6],
                                   [0.9, 0.9, 0.9, 0.9],
                                   [1.0, 1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0, 1.0] ])
        elif 'res50' in model_name and self.gpu_demand == 2:
             self.tput = np.array([[0.2, 0.2, 0.2, 0.2],
                                   [0.6, 0.6, 0.6, 0.6],
                                   [0.8, 0.8, 0.8, 0.8],
                                   [0.85, 0.85, 0.85, 0.85],
                                   [0.9, 0.9, 0.9, 0.9],
                                   [1.0, 1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0, 1.0] ])
        elif 'shuffle' in model_name:
             self.tput = np.array([[0.2, 0.2, 0.2, 0.2],
                                   [0.4, 0.4, 0.4, 0.4],
                                   [0.5, 0.5, 0.5, 0.5],
                                   [0.6, 0.6, 0.6, 0.6],
                                   [0.7, 0.7, 0.7, 0.7],
                                   [1.0, 1.0, 1.0, 1.0],
                                   [1.5, 1.5, 1.5, 1.5] ])
        elif 'mobile' in model_name:
             self.tput = np.array([[0.2, 0.2, 0.2, 0.2],
                                   [0.4, 0.4, 0.4, 0.4],
                                   [0.5, 0.5, 0.5, 0.5],
                                   [0.55, 0.55, 0.55, 0.55],
                                   [0.6, 0.6, 0.6, 0.6],
                                   [0.9, 0.9, 0.9, 0.9],
                                   [1.0, 1.0, 1.0, 1.0] ])
        elif self.gpu_demand == 1:
             self.tput = np.array([[1.0, 1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0, 1.0] ])
        else:
            self.tput = np.array([[0.6, 0.6, 0.6, 0.6],
                                   [1.0, 1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0, 1.0] ])

def parse_workload(trace):
    jobs = []
    with open(trace, 'r') as rf:
         lines = rf.readlines()
         for line in lines:
             params = line.strip().split(',')
             job = Job(job_id = int(params[0]),
                       name = params[1],
                       arrival = float(params[2]),
                       iters = int(params[3]),
                       gpu_demand = int(params[4]))
             jobs.append(job)
    return jobs

def generate_workload(gpus, multi=False):
    jobs = []
    job_id = 0
    arrival = 0.0
    workloads = ['alexnet', 'res18', 'shufflenet', 'mobilenet', 'res50', 'gnmt', 'transformer']
    gpu_demand = [1,2]
    gpus_allocated = 0
    print("gpus-alloc ={}, gpus={}".format(gpus_allocated, gpus))
    while gpus_allocated < gpus:
        model_name = random.choice(workloads)    
        iters = random.choice(range(1000, 3000))
        gpu = random.choice(gpu_demand)
        if not multi:
            gpu = 1

        if gpu > gpus - gpus_allocated:
            gpu = 1

        job = Job(job_id = job_id,
                  name = model_name,
                  arrival = round(arrival,2),
                  iters = iters,
                  gpu_demand = gpu)
      

        gpus_allocated += gpu
        job_id += 1
        arrival += 0.001
        jobs.append(job)
    with open('opt_record.log', 'w+') as f:
        for job in jobs:
            line = str(job.job_id) + "," + \
                   job.name + "," + \
                   str(job.arrival) + "," + \
                   str(job.iters) + "," + \
                   str(job.gpu_demand) + "\n"
            f.write(line)

    return jobs
   

def parser():
    parser = argparse.ArgumentParser(description='Parse Arguments.')     
    parser.add_argument('--trace', default=None, type=str)
    parser.add_argument('--solver', default=None, type=str)
    parser.add_argument('--solver2', default=None, type=str)
    parser.add_argument('--ilp1', action='store_true', default=False)
    parser.add_argument('--ilp2', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--phase1', action='store_true', default=False)
    parser.add_argument('--multi', action='store_true', default=False)
    parser.add_argument('--num_servers', default=1, type=int)
    parser.add_argument('--seed', default=10, type=int)
    args = parser.parse_args()
    return args 

def main():
    args = parser()
    logging.basicConfig()
    random.seed(args.seed)
    print(cp.installed_solvers())
    if args.trace is not None:
        jobs = parse_workload(args.trace)
    else:
        jobs = generate_workload(
                 args.num_servers*GPU_PER_SERVER, 
                 multi=args.multi)
    if args.debug:
        for job in jobs:
            print(str(job))
    start = time.time()
    cvxpy_solver = Solver(num_servers=args.num_servers, debug=args.debug)
    cvxpy_solver.SolveFirstLP(jobs, ilp=args.ilp1, solver=args.solver)
    end_p1 = time.time()
    if not args.phase1:
        cvxpy_solver.SolveSecondLP(ilp=args.ilp2, solver=args.solver2)
    end_p2 = time.time()
    if cvxpy_solver.is_fractional:
        cvxpy_solver.RoundRobin()
    end_p3 = time.time()
    print("Num jobs={}, multi-GPU={}, GPUs={}, servers={}".format(
                 len(jobs),
                 args.multi,
                 args.num_servers*GPU_PER_SERVER,
                 args.num_servers))
    print("Phase 1 time = {:.2f} s".format(end_p1-start))
    print("Phase 2 time = {:.2f} s".format(end_p2-end_p1))
    print("Phase 3 time = {:.2f} s".format(end_p3-end_p2))
    print("-"*75)
    effective_tput = 0
    serv_alloc_map = {}
    for i in range(args.num_servers):
        serv_alloc_map[i] = {}
        serv_alloc_map[i]['gpu'] = 0
        serv_alloc_map[i]['cpu'] = 0
        serv_alloc_map[i]['mem'] = 0
    print("{:<30} {:<8} {:<8} {:<8} {:<8}".format('Job', 'Server', 'GPU', 'CPU', 'Mem'))
    for job in jobs:
        print("-"*75) 
        #print("[{}]: {}".format(str(job), job.res_map))
        cpu_allocated = 0
        for serv, alloc in job.res_map.items():
            print("{:<30} {:<8} {:<8} {:<8} {:<8}".format(str(job), serv, alloc['gpu'], alloc['cpu'], alloc['mem']))
            cpu_allocated +=  alloc['cpu'] 
            serv_alloc_map[serv]['gpu'] += alloc['gpu']
            serv_alloc_map[serv]['cpu'] += alloc['cpu']
            serv_alloc_map[serv]['mem'] += alloc['mem']
        tput = job.tput[job.get_idx(job.cpu_val, cpu_allocated), 0]   
        effective_tput += tput
    print("-"*75) 
    print("\n\nFinal tput={}, Ideal tput={}".format(effective_tput.round(2), cvxpy_solver.tput.round(2)))
    print("-"*75) 
    print("{:<8} {:<8} {:<8} {:<8}".format('Server', 'GPU', 'CPU', 'Mem'))
    print("-"*75) 
    for serv, alloc in serv_alloc_map.items():
        print("{:<8} {:<8} {:<8} {:<8}".format(serv, alloc['gpu'], alloc['cpu'], alloc['mem']))
    print("-"*75) 
    
    print("\n{:.2f},{:.2f},{:.2f},{:.2f}".format(end_p1-start, end_p2-end_p1,end_p3-end_p2, end_p3-start))



if __name__ == "__main__":
    main()
