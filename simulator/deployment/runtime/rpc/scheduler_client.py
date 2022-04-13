#!/usr/bin/env python3

import os
import sys

import grpc

sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))
import common_pb2
import scheduler_to_worker_pb2 as s2w_pb2
import scheduler_to_worker_pb2_grpc as s2w_pb2_grpc
from google.protobuf.struct_pb2 import Struct


class JobDescription:
    """
    Description of how to run a job.
    """
    def __init__(self, job_id, command, work_dir=".",
            env={}):
        self.job_id = job_id
        self.command = command
        self.work_dir = work_dir
        self.env = env

class SchedulerRpcClient:
    """Scheduler client for sending RPC requests to a worker server."""

    def __init__(self, server_ip_addr, port):
        self._addr = server_ip_addr
        self._port = port
        self._server_loc = '{}:{}'.format(server_ip_addr, port)

    @property
    def addr(self):
        return self._addr

    @property
    def port(self):
        return self._port

    def run_job(self, job_description, round_num):
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)
            env = Struct()
            env.update(job_description.env)
            job_desc = s2w_pb2.JobDescription (\
                  job_id = job_description.job_id,
                  command = job_description.command,
                  work_dir = job_description.work_dir,
                  env = env)
            request = s2w_pb2.RunJobRequest(
                  job_description = job_desc, 
                  round_num = round_num)
            stub.RunJob(request)

    def kill_job(self, job_id):
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)
            request = s2w_pb2.KillJobRequest()
            request.job_id = job_id
            stub.KillJob(request)

    def reset(self):
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)
            stub.Reset(common_pb2.Empty())

    def shutdown(self):
        with grpc.insecure_channel(self._server_loc) as channel:
            stub = s2w_pb2_grpc.SchedulerToWorkerStub(channel)
            stub.Shutdown(common_pb2.Empty())
