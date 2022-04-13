#!/usr/bin/env python3

from concurrent import futures
import logging
import os
import sys
import socket
import time
import traceback

import grpc

sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import common_pb2
import job_to_scheduler_pb2 as j2s_pb2
import job_to_scheduler_pb2_grpc as j2s_pb2_grpc
import worker_to_scheduler_pb2 as w2s_pb2
import worker_to_scheduler_pb2_grpc as w2s_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class SchedulerWorkerRpcServer(w2s_pb2_grpc.WorkerToSchedulerServicer):
    def __init__(self, callbacks, logger):
        self._callbacks = callbacks
        self._logger = logger

    def RegisterWorker(self, request, context):
        try:
            round_duration, machine_id = self._callbacks["RegisterWorker"](\
                num_gpus=request.num_gpus,
                ip_addr=request.ip_addr,
                port=request.port)
            self._logger.info("Successfully registered worker {} with {} GPUs, round={}".format(
                machine_id, request.num_gpus, round_duration))
            return w2s_pb2.RegisterWorkerResponse(success=True, round_duration=round_duration, machine_id=machine_id)
        except Exception as e:
            self._logger.error("Register worker error: %s" % e)
            return w2s_pb2.RegisterWorkerResponse(success=False, round_duration=None, machine_id=None)

class SchedulerJobRpcServer(j2s_pb2_grpc.JobToSchedulerServicer):
    def __init__(self, callbacks, logger):
        self._callbacks = callbacks
        self._logger = logger

    def RegisterJob(self, request, context):
        try:
            (time_elapsed_this_round_s, round_duration_s) =\
                self._callbacks["RegisterJob"](request.job_id)
            self._logger.info("Successfully registered job %s" % request.job_id)
            return j2s_pb2.RegisterJobResponse(\
                time_elapsed_this_round_s=time_elapsed_this_round_s,\
                round_duration_s=round_duration_s)
        except Exception as e:
            self._logger.error("Register job error: %s" % e)
            return j2s_pb2.RegisterJobResponse(\
                time_elapsed_this_round_s=-1,\
                round_duration_s=-1)

    def UpdateIters(self, request, context):
        self._logger.info("Received update iters request from job {}".format(request.job_id))
        try:
            num_steps_per_round = self._callbacks["UpdateIters"](\
                request.job_id, request.num_steps,\
                request.execution_time_s, request.extend_lease, request.round_num)
        except Exception as e:
            self._logger.error("Update iters error in job %s: e" % (request.job_id, e))
            traceback.print_exc()
        return j2s_pb2.UpdateItersResponse(num_steps_per_round=num_steps_per_round)

    def RoundEnd(self, request, context):
        self._logger.info("Received round end notification from job %s" % request.job_id)
        terminate_lease = False
        try:
            terminate_lease = self._callbacks["RoundEnd"](\
                request.job_id, request.num_steps, request.execution_time_s, request.done)
        except Exception as e:
            self._logger.error("Round end error in job %s: %s" % (request.job_id, e))
            traceback.print_exc()
        return j2s_pb2.RoundEndResponse(terminate_lease=terminate_lease)

    def LeaseEnded(self, request, context):
        self._logger.info("Received lease ended notification from job %s" % request.job_id)
        try:
            self._callbacks["LeaseEnded"](request.job_id, request.num_steps, request.execution_time_s, request.round_num)
        except Exception as e:
            self._logger.error("Lease ended error in job %s: %s" % (request.job_id, e))
            traceback.print_exc()
        return common_pb2.Empty()

def serve(port, callbacks):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                      style='{'))
    logger.addHandler(ch)
    server = grpc.server(futures.ThreadPoolExecutor())
    w2s_pb2_grpc.add_WorkerToSchedulerServicer_to_server(
            SchedulerWorkerRpcServer(callbacks, logger), server)
    j2s_pb2_grpc.add_JobToSchedulerServicer_to_server(
            SchedulerJobRpcServer(callbacks, logger), server)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip_address = s.getsockname()[0]
    s.close()
    #ip_address = socket.gethostbyname(socket.gethostname())
    server.add_insecure_port('{}:{}'.format(ip_address, port))
    logger.info('Starting server at {0}:{1}'.format(ip_address, port))
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

# For testing purposes only
if __name__ == "__main__":
    def register_worker_dummy(num_gpus, ip_addr, port):
        return
    def register_job_dummy(job_id):
        return (5, 10)
    def update_iters_dummy(job_id, num_steps, execution_time_s, extend):
        return 18
    def round_end_dummy(job_id, num_steps, execution_time_s, done):
        return done
    def lease_ended_dummy(job_id):
        return
    dummy_callbacks = {
        "RegisterWorker": register_worker_dummy,
        "RegisterJob": register_job_dummy,
        "UpdateIters": update_iters_dummy,
        "RoundEnd": round_end_dummy,
        "LeaseEnded": lease_ended_dummy
    }
    serve(14040, dummy_callbacks)

