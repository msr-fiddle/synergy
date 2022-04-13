#!/usr/bin/env python3

import datetime
import json
import sys
import math

import numpy as np

from jobs.job import Job
from helpers.utils import get_gavel_like_iter, small_trace_dur
# Helper script to parse jobs from the philly trace into Job objects
# Trace: https://github.com/msr-fiddle/philly-traces

def format_time(t):
  return datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S")

def parse_jobs_full(path, sum_attempts=False, exponential=False, multigpu=False, debug_multi=False, small_trace=False, logger=None):
  """
  Parse the jobs described in the given path into Job objects.

  The schema is expected to match 'cluster_job_log' under
  https://github.com/msr-fiddle/philly-traces.

  If `sum_attempts` is True, this function will use the sum of all
  durations across a job's attempts as the job's duration. Otherwise,
  we will treat each attempt as a separate job where the arrival time
  of each attempt is the end time of the previous attempt.
  """
  print("Sum attempts  = {}, exponential = {}, multigpu = {}, debug_multi={}".format(sum_attempts,exponential,multigpu, debug_multi))
  with open(path) as f:
    data = json.load(f)
  # Filter out the jobs with zero attempts or with invalid end times
  data = [j for j in data if len(j["attempts"]) > 0]

  # First, find min submitted time, use this as t = 0
  first_submitted_time = min([format_time(j["submitted_time"]) for j in data])

  # Parse the jobs
  prev_gpu_demand = 0
  first_arrival = None
  jobs = []
  for j in data:
    submitted_time = format_time(j["submitted_time"])
    arrival_time_seconds = (submitted_time - first_submitted_time).total_seconds()
    cluster_id = j["vc"]

    duration_sum_seconds = 0
    previous_attempt_end_time = None
    queueing_delay = 0
    for i, a in enumerate(j["attempts"]):
      start_time = a["start_time"]
      end_time = a["end_time"]
      if "None" in start_time or "None" in end_time:
        continue
      
      duration_seconds = (format_time(end_time) - format_time(start_time)).total_seconds()
      gpu_demand = sum([len(d["gpus"]) for d in a["detail"]])

      # Max 16 GPU job
      if multigpu and gpu_demand > 16:
          continue

      if not exponential and multigpu and gpu_demand not in [1,2,4,8,16]:
          continue
          #gpu_demand = 1
      elif multigpu and gpu_demand not in [1,2,4,8,16]:
          gpu_demand = 1

      if duration_seconds <= 0.0 or gpu_demand <= 0:
        continue
    

      # If we treat all attempts of this job as the same job, sum the durations
      # Note: we assume the GPU demands are the same across all attempts
      if sum_attempts:
        duration_sum_seconds += duration_seconds
        if i == 0:
          queueing_delay = (format_time(start_time) - submitted_time).total_seconds()
      else:
        # Otherwise, we treat each attempt as its own job, in which case we
        # use the end time of the previous attempt as our arrival time
        if previous_attempt_end_time is not None:
          arrival_time_seconds = (previous_attempt_end_time - first_submitted_time).total_seconds()
          queueing_delay = (format_time(start_time) - previous_attempt_end_time).total_seconds()
        else:
          queueing_delay = (format_time(start_time) - submitted_time).total_seconds()
        previous_attempt_end_time = format_time(end_time)

      # Create the job, either once per attempt or only on the last attempt if
      # we are summing across jobs
      if not sum_attempts or i == len(j["attempts"]) - 1:
        
        # gavel/gandiva like workload
        if duration_sum_seconds <= math.pow(10, 1.5) * 60 or\
          duration_sum_seconds >= math.pow(10, 4) * 60:
          continue

        job_id = j["jobid"]

        #logger.info(j)
        #logger.info("Arr = {}, dur = {}, q = {}".format(arrival_time_seconds, duration_sum_seconds, queueing_delay))

        if not sum_attempts:
          job_id += "-attempt%d" % (i + 1)

        # For now, assume all job iterations are 1 second
        # Will set iterations during model selction
        iteration_time_seconds = 1

        if small_trace:
          total_iterations = small_trace_dur()
        elif exponential:
          total_iterations = get_gavel_like_iter()
        else:
          total_iterations = duration_sum_seconds if sum_attempts else duration_seconds

        # TODO: Fill in packing and placement scores later
        # There are too many jobs (100k+) to build a full matrix for each job
        packing_scores = None
        placement_scores = None
        synergy_res_scores = None
        synergy_storage_scores = None

        if not multigpu:
            gpu_demand = 1
        # TODO DEBUG : For synergy opt verif
        # elif gpu_demand >2 :
        #    gpu_demand = 2

        if debug_multi:
            prev_gpu_demand = max(1, (prev_gpu_demand + 1) % 8)
            if prev_gpu_demand % 2 == 1 and prev_gpu_demand > 1:
                prev_gpu_demand += 1
            gpu_demand =  prev_gpu_demand 

        #verify if trace is parsed correctly
        if sum_attempts and not exponential:
            if not multigpu:
               assert gpu_demand == 1, "Not single-GPU workload"

            assert total_iterations == duration_sum_seconds, "Job duration doesnt match trace"

        # Get jobs only in this range where demand > 512 GPUs
        if not exponential:
           if arrival_time_seconds <= 3600*2400 or \
               arrival_time_seconds >= 3600*3000:
               continue



        jobs.append(Job(
          job_id,
          arrival_time_seconds,
          iteration_time_seconds,
          total_iterations,
          #1,
          gpu_demand,
          packing_scores,
          placement_scores,
          synergy_res_scores,
          synergy_storage_scores,
          j["user"],
          job_queueing_delay=queueing_delay,
          cluster_id=cluster_id,
          iter_is_duration=True))

        if not exponential and len(jobs) == 8000:
            jobs.sort(key=lambda job: job.job_arrival_time)
            first_arrival = jobs[0].job_arrival_time

            for job in jobs:
                job.job_arrival_time -= first_arrival
                logger.info("Job : {}, arr : {:.2f}, iter_time : {}, iters : {}, gpu : {}\n".format(job.job_id, job.job_arrival_time/3600, job.job_iteration_time, job.job_total_iteration, job.job_gpu_demand))
            return jobs

  if not exponential:
      jobs.sort(key=lambda job: job.job_arrival_time)
  #for job in jobs:
  #   logger.info("Job : {}, arr : {:.2f}, iter_time : {}, iters : {}, gpu : {}\n".format(job.job_id, job.job_arrival_time/3600, job.job_iteration_time, job.job_total_iteration, job.job_gpu_demand))
  #sys.exit(1)
  return jobs

if __name__ == "__main__":
  args = sys.argv
  if len(args) != 2:
    print("Usage: ./parse_philly_jobs.py [path_to_job_trace]")
    sys.exit(1)
  parse_jobs(args[1])
