import os
import sys
import cvxpy as cp
import numpy as np
import logging
import argparse
import random
import time
import copy
import math

from jobs.workload import Workload

GPU_PER_SERVER=8

class Solver():
    def __init__(
        self,
        num_servers=1,
        cpu_per_server=24,
        mem_per_server=500,
        gpu_per_server=GPU_PER_SERVER,
        debug=False,
        round_dur=300):
        
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
        self.round_dur = round_dur
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
    
    # tput matrix gives normalized tput for varying cpu and memory
    # PER GPU
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
             cpu_constraint = cpu_constraint + cp.sum(cp.multiply(cpu_val_trans, self.Y[job.job_id]))*job.job_gpu_demand
             #print(cpu_val_trans, cpu_val_trans.shape)
        self.constraints.append(cpu_constraint <= self.cpus)


        mem_constraint = 0
        for job in jobs:
             mem_val = np.array([list(job.mem_val.values())])
             mem_constraint = mem_constraint + cp.sum(cp.multiply(mem_val, self.Y[job.job_id]))* job.job_gpu_demand
        self.constraints.append(mem_constraint <= self.mem)

        config_constraint = 0
        for job in jobs:
             config_constraint = cp.sum(self.Y[job.job_id])
             self.constraints.append(config_constraint == 1)

        fair_share_constraint = 0
        for job in jobs:
             fair_share_constraint = cp.sum(cp.multiply(job.tput, self.Y[job.job_id]))
             fair_cpu_idx = job.get_idx(job.cpu_val, self.fair_cpu)
             fair_mem_idx = job.get_idx(job.mem_val, self.fair_mem)
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
        #print(cvxprob.value)
        self.tput = cvxprob.value
        #print(cvxprob.status)
        cpu_list = []
        mem_list = []
        #if self.debug:
        for job in jobs:
             #print("--"*20)
             #print(str(job))
             cpu_val = np.array([list(job.cpu_val.values())])
             cpu_val_trans = np.transpose(cpu_val)
             mem_val = np.array([list(job.mem_val.values())])
             mem = cp.sum(cp.multiply(mem_val, self.Y[job.job_id]))*job.job_gpu_demand
             cpu = cp.sum(cp.multiply(cpu_val_trans, self.Y[job.job_id]))* job.job_gpu_demand
             cpu_list.append(cpu.value.round(2))
             mem_list.append(mem.value.round(2))
             fair_share_constraint = cp.sum(cp.multiply(job.tput, self.Y[job.job_id]))
             fair_cpu_idx = job.get_idx(job.cpu_val, self.fair_cpu)
             fair_mem_idx = job.get_idx(job.mem_val, self.fair_mem)
             if fair_cpu_idx is None or fair_mem_idx is None:
                 raise ValueError("Invalid idx in tput map")

             fair_share = job.tput[fair_cpu_idx][fair_mem_idx]
             #print("\tEstimated tput={}, Fair share tput ={}".format(fair_share_constraint.value.round(2), fair_share.round(2)))
             #print("\tcpu= {}, mem = {}, fair_mem={}".format(cpu.value, mem.value, fair_mem_idx))

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
                mem = cp.sum(cp.multiply(mem_val, self.Y[job.job_id]))*job.job_gpu_demand
                cpu = cp.sum(cp.multiply(cpu_val_trans, self.Y[job.job_id]))*job.job_gpu_demand
                cpu_list.append(cpu.value.round(2))
                mem_list.append(mem.value.round(2))
                fair_share_constraint = cp.sum(cp.multiply(job.tput, self.Y[job.job_id]))
                fair_cpu_idx = job.get_idx(job.cpu_val, self.fair_cpu)
                fair_mem_idx = job.get_idx(job.mem_val, self.fair_mem)
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
            gpu_demand_vec.append(job.job_gpu_demand)
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

                    job.res_map[server_id]['gpu'] = (server_alloc*job.job_gpu_demand).round(2)
                    job.res_map[server_id]['cpu'] = (server_alloc*job.cpu_demand).round(2)
                    job.res_map[server_id]['mem'] = (server_alloc*job.mem_demand).round(2)

            print("Job :{}, Res_map={}".format(str(job), job.res_map)) 
        print("-"*45)


    def allocate_round(self, job):
        norm_tput = job.tput[
         job.get_idx(job.cpu_val, job.cpu_demand/job.job_gpu_demand), 
         job.get_idx(job.mem_val, job.mem_demand/job.job_gpu_demand)]
        fair_cpu_idx = job.get_idx(job.cpu_val, self.fair_cpu)
        fair_mem_idx = job.get_idx(job.mem_val, self.fair_mem)
        fair_tput = job.tput[fair_cpu_idx, fair_mem_idx]
        iter_time = job.job_iteration_time / norm_tput
        num_iters_this_round = math.floor(self.round_dur / iter_time)
        job.job_executed_iteration += num_iters_this_round
        if job.job_executed_iteration > job.job_total_iteration:
            job.job_executed_iteration = job.job_total_iteration
        job.attained_service_time += self.round_dur
        print("Fair tput ={}, current={}, new iter={}".format(fair_tput, norm_tput, iter_time))
        

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
            job_demand_vec['gpu'] = job.job_gpu_demand
            #job_demand_vec['cpu'] = self.fair_cpu * job.gpu_demand
            job_demand_vec['cpu'] = min(self.fair_cpu * job.job_gpu_demand, job.cpu_demand)
            #job_demand_vec['mem'] = self.fair_mem * job.gpu_demand
            job_demand_vec['mem'] = min(self.fair_mem * job.job_gpu_demand, job.mem_demand)
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
            job_demand_vec['gpu'] = job.job_gpu_demand
            job_demand_vec['cpu'] = job.job_gpu_demand
            job_demand_vec['mem'] = job.job_gpu_demand*25
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
    end = time.time()
    if not args.phase1:
        cvxpy_solver.SolveSecondLP(ilp=args.ilp2, solver=args.solver2)
    end_p2 = time.time()
    if cvxpy_solver.is_fractional:
        cvxpy_solver.RoundRobin()
    print("Num jobs={}, multi-GPU={}, GPUs={}, servers={}".format(
                 len(jobs),
                 args.multi,
                 args.num_servers*GPU_PER_SERVER,
                 args.num_servers))
    print("Phase 1 time = {:.2f} s".format(end-start))
    print("Phase 2 time = {:.2f} s".format(end_p2-end))
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
    

if __name__ == "__main__":
    main()
