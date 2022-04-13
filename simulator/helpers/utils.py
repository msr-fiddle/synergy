import math
import random
import numpy as np
#from resources.server_config import Res

def poisson_next_arrival_time(jobs_per_hour):
    next_arrival_in_hours = -math.log(1.0 - random.random()) / jobs_per_hour 
    next_arrival_in_seconds = math.ceil(next_arrival_in_hours * 60 * 60)
    return next_arrival_in_seconds

def get_total_iteration(range_min, range_max):
    return random.randint(range_min, range_max)

def get_job_gpu_demand():
    rand_var = random.uniform(0,1)
    if rand_var >= 0.95:
        return 8
    elif 0.8 <= rand_var < 0.95:
        return 4
    elif 0.7 <= rand_var < 0.8:
        return 2
    else:
        return 1
    

def exponential(lambd):
    while True:
        yield random.expovariate(lambd)

def get_total_iteration_exp(range_min, range_max):
    # lambda = 1
    new_range = range_max - range_min
    
    #x = np.random.exponential()   
    # lambd = 1/(mean) 
    #x = math.ceil(random.expovariate(2/new_range))
    #if x < 0:
    #    print(x, 2/new_range)
    #if x < range_min:
    #    x = x + range_min
    #if x > range_max:
    #    x = range_max
    #rand_num = 1 - math.exp(-x)
    #mean = 100K iters = 27hrs
    x = 0
    while not range_min <= x <= range_max:
        x = math.ceil(random.expovariate(1/60000))
    return x

def get_gavel_like_iter():
    if random.random() >= 0.8:
        iters = 60 * (10 ** random.uniform(3,4))
    else:
        iters = 60 * (10 ** random.uniform(1.5,3))
    return iters

def small_trace_dur():
    # 
    if random.random() >= 0.8:
        iters = 60 * (10 ** random.uniform(2.5,2.7))
    else:
        iters = 60 * (10 ** random.uniform(1.5,2.5))
    return iters
        
def gpu_normalized_vector(vector):
    return [item/vector[0] for item in vector]

def cumulative_map(orig_map, new_map):
   """
   Adds the entries of new map into orig and returns orig
   """
   for key, value in new_map.items():
       if key in orig_map:
           orig_map[key] += value
       else:
           orig_map[key] = value

   return orig_map
