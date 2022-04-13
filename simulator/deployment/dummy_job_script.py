import sys
import os
import time
import traceback

#from runtime.rpc import scheduler_client, scheduler_server
from synergy_iterator import SynergyIterator

#Dummy scheduler code which registers workers based on cluster config


def main():
    dummy_dl = list(range(50))
    synergy_iterator = SynergyIterator(dummy_dl, mock=True)
    for i,_ in enumerate(synergy_iterator):
        """
        Assume iter durations are 300 ms each
        """
        time.sleep(0.3)
        print("Processing iteration {}/{} : R {} : Steps : {}/{}".format(i+1, len(dummy_dl), synergy_iterator._round, synergy_iterator._steps_this_round, synergy_iterator._num_steps_per_round)) 
    synergy_iterator.complete()
    print("Job complete!")
    


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 4:
        print("Usage python job_script.py [sched ip] [sched port] [job_id]")
        sys.exit(1)

    sched_addr, sched_port, job_id = args[1], args[2], args[3]
    os.environ["SYNERGY_JOB_ID"] = str(job_id)    
    os.environ["SYNERGY_WORKER_ID"] = "234" 
    os.environ["SYNERGY_SCHED_ADDR"] = sched_addr
    os.environ["SYNERGY_SCHED_PORT"] = sched_port 
    os.environ["SYNERGY_LOG_DIR"] = "." 
    os.environ["SYNERGY_DEBUG"] = "true" 
   
    main()   

