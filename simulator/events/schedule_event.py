from events.event_type import EventType
from events.cluster_event import ClusterEvent
from tput import update_tput, get_idx
import sys

class ScheduleEvent(ClusterEvent):

    def __init__(self, time, scheduler):
        super().__init__(
            time, 
            int(EventType.SCHEDULE))
        self.scheduler = scheduler

    def handleEvent(self):
        super().handleEvent()
        runner = ClusterEvent.runner
        jobs = runner.runnable_jobs
        gpus = runner.cluster.get_free_gpus()
        
        #self.logger.info("\nGPU status : Free GPUs = {} ".format(len(gpus)))
        #self.logger.info("\nJob status : ")
        #for job in jobs:
        #    self.logger.info("\t{}".format(str(job)))
        self.scheduler.schedule(jobs, gpus)

        """
        print("Job schedule")
        # Only for comparison
        #-------------------------------------------------
        serv_alloc_map = {}
        effective_tput = 0
        for i in runner.cluster.servers:
            serv_alloc_map[i] = {}
            serv_alloc_map[i]['gpu'] = 0
            serv_alloc_map[i]['cpu'] = 0
            serv_alloc_map[i]['mem'] = 0

        print("-"*100)
        print("{:<50} {:<8} {:<8} {:<8} {:<8}".format('Job', 'Server', 'GPU', 'CPU', 'Mem'))
        self.scheduler.running_jobs.sort(
            key=lambda job:
            (job.job_id))
        for job in self.scheduler.running_jobs:
            #gpu_ids = [gpu.gpu_id for gpu in job.gpus]
            #self.logger.info("Job :{}, GPU:{}, CPU:{}, mem:{}, res_map:{}".format(str(job), gpu_ids, job.cpus, job.mem, job.res_map))
           print("-"*100)
           update_tput(job, job.job_model.model_name)
           cpu_allocated = 0
           for serv, alloc in job.res_map.items():
               print("{:<50} {:<8} {:<8} {:<8} {:<8}".format(str(job), serv.server_id, alloc['gpu'], alloc['cpu'], alloc['mem']))
               cpu_allocated +=  alloc['cpu']
               serv_alloc_map[serv]['gpu'] += alloc['gpu']
               serv_alloc_map[serv]['cpu'] += alloc['cpu']
               serv_alloc_map[serv]['mem'] += alloc['mem']
           tput = job.tput[get_idx(cpu_allocated), 0]
           #self.logger.info("[{}]: {}, tput={}".format(str(job), job.res_map, tput))
           effective_tput += tput
        print("-"*100)
        print("Final tput = {}".format(effective_tput))
        print("-"*75)
        
        print("{:<8} {:<8} {:<8} {:<8}".format('Server', 'GPU', 'CPU', 'Mem'))
        print("-"*75)
        for serv, alloc in serv_alloc_map.items():
            print("{:<8} {:<8} {:<8} {:<8}".format(serv.server_id, alloc['gpu'], alloc['cpu'], alloc['mem']))
        print("-"*75)
        #print("Final server map = \n{}".format(serv_alloc_map))
        sys.exit(0)           
        #-------------------------------------------------
        """


        if not runner.simulate:
            self.scheduler.deploy_jobs_round()

        next_round_time = runner.get_time() + self.scheduler.round_duration



        if runner.simulate:
            if runner.static and len(runner.runnable_jobs) == 0:
                runner.terminate = True
            elif runner.trace is not None and len(runner.runnable_jobs) == 0:
                runner.terminate = True
            else:
                runner.add_event(ScheduleEvent(next_round_time, self.scheduler))
        else:
            #cluster run termination condition
            #runner.add_event(ScheduleEvent(next_round_time, self.scheduler))
            if len(runner.runnable_jobs) == 0:
                runner.terminate =  True


        # profile cluster allocation + utilization
        alloc_map = runner.cluster.alloc_stats
        runner.cluster_alloc.put(alloc_map, runner.time) 
        util_map = runner.cluster.utilization_stats
        runner.cluster_util.put(util_map, runner.time) 

        demand_map = runner.cluster.demand_stats
        runner.cluster_demand.put(demand_map, runner.time) 
#        max_resources = [runner.max_gpus, runner.max_cpus, runner.max_mem]
#        aggregate = np.zeros(3)
#        for job in self.scheduler.running_jobs:
#            aggregate[0] += job.job_gpu_demand
#            aggregate[1] += job.job_cpu_demand
#            aggregate[2] += job.job_mem_demand

        # Get % demand
#        aggregate = aggregate/max_resources*100
#        runner.cluster_demand.put_aggregate(aggregate, runner_time)
