#!/usr/bin/env python3

import logging
import os
import sys
import time

import grpc

sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))
import worker_to_scheduler_pb2 as w2s_pb2
import worker_to_scheduler_pb2_grpc as w2s_pb2_grpc

LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class WorkerRpcClient:
    """Worker client for sending RPC requests to a scheduler server."""

    def __init__(self, worker_ip_addr, worker_port,
                 sched_ip_addr, sched_port):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                          style='{'))
        logger.addHandler(ch)
        self._logger = logger
        self._worker_ip_addr = worker_ip_addr
        self._worker_port = worker_port
        self._sched_loc = '{}:{}'.format(sched_ip_addr, sched_port)

    def register_worker(self, num_gpus):
        request = w2s_pb2.RegisterWorkerRequest(
            num_gpus=num_gpus,
            ip_addr=self._worker_ip_addr,
            port=self._worker_port)
        with grpc.insecure_channel(self._sched_loc) as channel:
            self._logger.info("Trying to register worker...")
            stub = w2s_pb2_grpc.WorkerToSchedulerStub(channel)
            response = stub.RegisterWorker(request)
            self._logger.info("Trying to register worker : Got response {}...".format(response.success))
            if response.success:
                self._logger.info("Successfully registered worker client to scheduler")
                return (response.success, response.round_duration, response.machine_id)
            else:
                self._logger.error('Failed to register worker!')
               
                return (response.success, None, None)

 
