import os


def _update_free_resources(job_demand_vector):
    alloc_map = {}
    alloc_map["gpu"] = job_demand_vector[0]        
    alloc_map["cpu"] = job_demand_vector[1]
    alloc_map["mem"] = job_demand_vector[2]
    alloc_map["sspeed"] = job_demand_vector[3]
    return alloc_map

def _fits_in_server(server_free_vector, required_res_vector):
    for idx, free_res in enumerate(server_free_vector):
        required_res = required_res_vector[idx]
        if free_res < required_res:
            return False
    return True
