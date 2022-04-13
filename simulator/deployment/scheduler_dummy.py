import sys

import os
import copy
import time
import traceback
import logging
import faulthandler
import socket
import subprocess

import queue
from concurrent.futures import ThreadPoolExecutor
import threading

from runtime.rpc import scheduler_client, scheduler_server
from synergy_iterator import SynergyIterator
from runtime.rpc.scheduler_client import JobDescription

#Dummy scheduler code which registers workers based on cluster config

# Port for scheduler server, IP will be local 
SCHEDULER_PORT = 14041


def get_self_ip():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address


class Scheduler:

    def __init__(self,
                 simulation = False,
                 round_duration = 360):


        # If set, runs in simulation mode without RPCs
        self.simulation = simulation

        self.logger = logging.getLogger(__name__)

        self.round_duration = round_duration
        self.job_completion_times = {}
       
        # List of IDs of servers in the cluster
        self.gpu_ids = []

        # Worker ID to assign
        self.gpu_id_counter = 0
        
        # RPC clients for each worker
        self.gpu_rpc_clients = {}
        self.all_rpc_clients = []

        # Scheduler synchronization primitives
        self.scheduler_lock = threading.Lock()
        self.scheduler_cv = threading.Condition(self.scheduler_lock)



        port = SCHEDULER_PORT
        callbacks = {
            'RegisterWorker': self.register_worker_callback,
            'RegisterJob' : self.register_job_callback,
            'UpdateIters' : self.update_iters_callback,
            'RoundEnd' : self.round_end_callback,
            'LeaseEnded' : self.lease_ended_callback
        }

        if not self.simulation:
            faulthandler.enable()
            f = open('.stack_trace.log', 'w')
            faulthandler.dump_traceback_later(30, repeat=True, file=f, exit=False)
         
            # Allocate and Dispatch jobs
            self.allocation_thread = threading.Thread(target=self.allocation_thread_fn)
            self.allocation_thread.daemon = True
            self.allocation_thread.start()


            # Scheduler Server
            self.server_thread = threading.Thread(target=scheduler_server.serve, 
                                                  args=(port, callbacks))
            self.server_thread.daemon = True
            self.server_thread.start()


            # Run scheduling decision for the next round
            self.mechanism_thread = threading.Thread(target=self.mechanism_thread_fn)
            self.mechanism_thread.daemon = True
            self.mechanism_thread.start()



    def mechanism_thread_fn(self):
        pass


    def allocation_thread_fn(self):
        pass

    """
    Registers a worker(GPUs in a server) with the scheduler.
   
    Registers a worker (ip, port) with `num_gpus` GPUs
    Assigns the worker an ID the same way as in simulation
    An RPC client is set up using the given (IP, Port) to enable
    communication between the scheduler and the worker to dispatch/kill jobs.
    
    Returns the IDs assigned to GPUs in the server
    """

    def register_worker_callback(self, num_gpus, ip_addr, port): 
        rpc_client = scheduler_client.SchedulerRpcClient(ip_addr, port)
        self.all_rpc_clients.append(rpc_client)

        with self.scheduler_lock:
            per_gpu_ids = []
            for i in range(num_gpus):
                gpu_id = self.gpu_id_counter
                per_gpu_ids.append(gpu_id)
                self.gpu_ids.append(gpu_id)
                self.gpu_id_counter += 1
                self.gpu_rpc_clients[gpu_id] = rpc_client
            self.scheduler_cv.notifyAll()
        return self.round_duration


    def register_job_callback(self, job_id):
        return (5,10)

    def update_iters_callback(self, job_id, num_steps, execution_time_s, extend):
        #print("returning 18")
        return 18

    def round_end_callback(self, job_id, num_steps, execution_time_s, done):
        return done

    def lease_ended_callback(self, job_id):
        return



def main(worker_addr, worker_port):
    scheduler = Scheduler()

    # Init worker server here for now. Must be done in worker seperately later
    new_env = copy.deepcopy(os.environ)
    cmd = "python launch_worker_server.py -i {} -s {} -w {} -g 8 --data_dir /datadrive/mnt4/jaya/datasets/ --checkpoint_dir ./chk/ --run_dir ./ --use_mps ".format(str(get_self_ip()), SCHEDULER_PORT, worker_port)
    print("Executing cmd {}".format(cmd))
    err_file = open("error.log", "wb")
    proc = subprocess.Popen(cmd,
                         #stdout=subprocess.PIPE,
                         #stderr=subprocess.STDOUT,
                         #stderr=err_file,
                         cwd="./",
                         env=new_env,
                         shell=True)
    #stdout, stderr = proc.communicate()
    #print(proc.stdout.decode('utf-8').strip()) 
    print("Done registering worker. Must sleep now")
    time.sleep(10)
    #sys.exit(1)
    scheduler.register_worker_callback(8, worker_addr, worker_port)
    
    job_id = 111
    #new_env = copy.deepcopy(os.environ)
    cmd = "python dummy_job_script.py " + str(get_self_ip()) + " " + str(SCHEDULER_PORT) + " " + str(job_id) 
    job_description = JobDescription(job_id, cmd, work_dir="./",env=new_env, gpu_indices=[str(0), str(1)], 
                       ranks=[str(0)])
    rpc_client = scheduler.gpu_rpc_clients[0] 
    round_num = 0
    rpc_client.run_job(job_description, round_num)
    #time.sleep(10)
    print("Returned from job run and waiting")
    #scheduler.server_thread.join()
    proc.wait()
    #proc.terminate()

if __name__ == "__main__":
    args = sys.argv
    if len(args) != 3:
        print("Usage python scheduler.py [worker ip] [worker port] [sched ip] [sched port]")
        sys.exit(1)

    worker_addr, worker_port = args[1], int(args[2])
    main(worker_addr, worker_port)
        

