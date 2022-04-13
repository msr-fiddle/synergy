#!/usr/bin/env python3

import os
import sys

import grpc

sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))
import job_to_scheduler_pb2 as j2s_pb2
import job_to_scheduler_pb2_grpc as j2s_pb2_grpc

class JobRpcClient:
    """Job client for sending RPC requests to a scheduler server."""

    def __init__(self, job_id, sched_ip_addr, sched_port):
        self._job_id = job_id
        self._sched_loc = "{}:{}".format(sched_ip_addr, sched_port)

    def register_job(self):
        request = j2s_pb2.RegisterJobRequest(job_id=self._job_id)
        with grpc.insecure_channel(self._sched_loc) as channel:
            stub = j2s_pb2_grpc.JobToSchedulerStub(channel)
            response = stub.RegisterJob(request)
            return (response.time_elapsed_this_round_s,\
                response.round_duration_s)

    def update_iters(self, num_steps, execution_time_s, extend, round_num):
        request = j2s_pb2.UpdateItersRequest(\
            job_id=self._job_id,
            num_steps=num_steps,
            execution_time_s=execution_time_s,
            extend_lease=extend,
            round_num=round_num)
        with grpc.insecure_channel(self._sched_loc) as channel:
            stub = j2s_pb2_grpc.JobToSchedulerStub(channel)
            response = stub.UpdateIters(request)
            return response.num_steps_per_round

    def round_end(self, num_steps, execution_time_s, done):
        request = j2s_pb2.RoundEndRequest(\
            job_id=self._job_id,
            num_steps=num_steps,
            execution_time_s=execution_time_s,
            done=done)
        with grpc.insecure_channel(self._sched_loc) as channel:
            stub = j2s_pb2_grpc.JobToSchedulerStub(channel)
            response = stub.RoundEnd(request)
            return response.terminate_lease

    def lease_ended(self, num_steps, execution_time_s, round_num):
        request = j2s_pb2.LeaseEndedRequest(\
            job_id=self._job_id, 
            num_steps=num_steps, 
            execution_time_s=execution_time_s,
            round_num=round_num)
        with grpc.insecure_channel(self._sched_loc) as channel:
            stub = j2s_pb2_grpc.JobToSchedulerStub(channel)
            stub.LeaseEnded(request)


