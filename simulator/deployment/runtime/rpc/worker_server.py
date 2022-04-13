#!/usr/bin/env python3

from concurrent import futures
import logging
import os
import socket
import subprocess
import sys
import threading
import time

import grpc

sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))
import common_pb2
import scheduler_to_worker_pb2 as s2w_pb2
import scheduler_to_worker_pb2_grpc as s2w_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class WorkerServer(s2w_pb2_grpc.SchedulerToWorkerServicer):
    def __init__(self, callbacks, condition, logger):
        self._callbacks = callbacks
        self._condition = condition
        self._logger = logger 

    def RunJob(self, request, context):
        job_id = request.job_description.job_id
        #job_ids = [jd.job_id for jd in request.job_descriptions]
        self._logger.debug("Received request to run job %s" % job_id)
        run_job_callback = self._callbacks['RunJob']
        jd = request.job_description
        run_job_callback(jd, request.round_num)
        return common_pb2.Empty()

    def KillJob(self, request, context):
        self._logger.debug("Received request to kill job %s" % request.job_id)
        kill_job_callback = self._callbacks['KillJob']
        kill_job_callback(request.job_id)
        return common_pb2.Empty()

    def Reset(self, request, context):
        self._logger.debug("Received reset request from the scheduler")
        reset_callback = self._callbacks['Reset']
        reset_callback()
        return common_pb2.Empty()

    def Shutdown(self, request, context):
        self._logger.debug("Received shutdown request the scheduler")
        # Handle any custom cleanup in the scheduler.
        shutdown_callback = self._callbacks['Shutdown']
        shutdown_callback()

        # Indicate to the worker server that a shutdown RPC has been received.
        self._condition.acquire()
        self._condition.notify()
        self._condition.release()

        return common_pb2.Empty()

def serve(port, callbacks):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                      style='{'))
    logger.addHandler(ch)
    condition = threading.Condition()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    s2w_pb2_grpc.add_SchedulerToWorkerServicer_to_server(
            WorkerServer(callbacks, condition, logger), server)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    s.connect(("8.8.8.8", 80))
    ip_address = s.getsockname()[0]
    s.close()
    #ip_address = socket.gethostbyname(socket.gethostname())
    server.add_insecure_port('{}:{}'.format(ip_address, port))
    logger.info('Starting server at {0}:{1}'.format(ip_address, port))
    server.start()

    # Wait for worker server to receive a shutdown RPC from scheduler.
    with condition:
        condition.wait()

    # Wait for shutdown message to be sent to scheduler.
    time.sleep(5)

# For testing purposes only
if __name__ == "__main__":
    def run_cmd(cmd, work_dir=None, env={}):
        print("Running command  in worker : '%s'" % cmd)
        print("Current working dir worker : '%s'" % os.getcwd())
        proc = subprocess.run(cmd,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              cwd=work_dir,
                              env=env,
                              shell=True)
        print(proc.stdout.decode('utf-8').strip())
    dummy_callbacks = {
        "RunJob": lambda jid, cmd, work_dir, env: run_cmd(cmd, work_dir, env),
        "KillJob": lambda jid: None,
        "Reset": lambda: None,
        "Shutdown": lambda: None
    }
    serve(18888, dummy_callbacks)

