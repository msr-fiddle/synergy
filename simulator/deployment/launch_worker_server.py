import argparse
import datetime
import logging
import os
import queue
import shutil
import socket
import signal
import sys
import threading
import subprocess

from runtime.rpc import job_launcher
from runtime.rpc import worker_client
from runtime.rpc import worker_server

import helper

CHECKPOINT_DIR_NAME = 'checkpoints'
LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class Worker:
    def __init__(self, sched_addr, sched_port, worker_port,
                 num_gpus, run_dir, data_dir, checkpoint_dir, use_mps):
        logger = logging.getLogger('worker')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT,
                                          style='{'))
        logger.addHandler(ch)
        self._logger = logger
        self._logging_handler = ch
        self._machine_id = None

        self.fh = open('worker.log', 'w+')

        num_available_gpus = helper.get_num_gpus()
        if num_gpus > num_available_gpus:
            raise ValueError('%d GPUs requested active, but only %d total '
                             'GPUs are available' % (num_gpus,
                                                     num_available_gpus))
        signal.signal(signal.SIGINT, self._signal_handler)
        self._gpu_ids = list(range(num_gpus))
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        self._worker_addr = s.getsockname()[0]
        s.close()
        #self._worker_addr = socket.gethostbyname(socket.gethostname())
        self._worker_port = worker_port
        self._worker_rpc_client = worker_client.WorkerRpcClient(
                self._worker_addr,
                self._worker_port, sched_addr, sched_port)

        callbacks = {
            'RunJob': self._run_job_callback,
            'KillJob': self._kill_job_callback,
            'Reset': self._reset_callback,
            'Shutdown': self._shutdown_callback,
        }
        def run_cmd(cmd, work_dir=None, env={}):
            print("Running command in w : '%s'" % cmd)
            print("Current working dir in w :  '%s'" % os.getcwd())
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

        print("Starting worker")

        self._server_thread = threading.Thread(
            target=worker_server.serve,
            args=(worker_port, callbacks,))
        self._server_thread.daemon = True
        self._server_thread.start()

        success, self._round_duration, self._machine_id = \
            self._worker_rpc_client.register_worker(num_gpus)
        if not success:
            raise RuntimeError("Error registering worker")

        
        if not os.path.isdir(checkpoint_dir):
            # Set up a new checkpoint directory if does not already exist.
            os.mkdir(checkpoint_dir)
        else:
            # Clear the checkpoints if they have already been created.
            for dirname in os.listdir(checkpoint_dir):
                if os.path.isdir(os.path.join(checkpoint_dir, dirname)):
                    shutil.rmtree(os.path.join(checkpoint_dir, dirname))

        self._job_launcher = job_launcher.JobLauncher(
                                                 self._machine_id,
                                                 self._round_duration,
                                                 self._gpu_ids,
                                                 self._worker_rpc_client,
                                                 sched_addr,
                                                 sched_port,
                                                 run_dir,
                                                 data_dir,
                                                 checkpoint_dir,
                                                 use_mps=use_mps)

        self.fh.write("Waiting to join worker server")
        self._server_thread.join()
        self.fh.write("Exiting worker server")

    def _run_job_callback(self, job_description, round_num):
        # hack to prevent a job being dispatched before the launcher is set up
        # TODO: fix this by sending a "I'm ready" message to scheduler
        while True:
            try:
                self._job_launcher
                break
            except Exception as e:
              continue
        self._logger.debug('Dispatching run request')
        self._job_launcher.dispatch_job(job_description, round_num)

    def _kill_job_callback(self, job_id):
        self._job_launcher._kill_jobs(job_id=job_id)

    def _signal_handler(self, sig, frame):
        self._job_launcher.shutdown()
        self._logger.removeHandler(self._logging_handler)
        self._logging_handler.close()
        sys.exit(0)

    def _reset_callback(self):
        self._job_launcher.reset()

    def _shutdown_callback(self):
        self._job_launcher.shutdown()
        self._logger.removeHandler(self._logging_handler)
        self._logging_handler.close()

    def join(self):
        self._server_thread.join()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run a worker process')
    parser.add_argument('-i', '--ip_addr', type=str, required=True,
                        help='IP address for scheduler server')
    parser.add_argument('-s', '--sched_port', type=int, default=50060,
                        help='Port number for scheduler server')
    parser.add_argument('-w', '--worker_port', type=int, default=50061,
                        help='Port number for worker server')
    parser.add_argument('-g', '--num_gpus', type=int, required=True,
                        help='Number of available GPUs')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Directory to run jobs from')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory where data is stored')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory where checkpoints is stored')
    parser.add_argument('--use_mps', action='store_true', default=False,
                        help='If set, enable CUDA MPS')
    args = parser.parse_args()
    opt_dict = vars(args)

    print(opt_dict)

    # TODO: Just pass args directly to maintain consistency with other code.
    worker = Worker(opt_dict['ip_addr'],
                    opt_dict['sched_port'], opt_dict['worker_port'],
                    opt_dict['num_gpus'], opt_dict['run_dir'],
                    opt_dict['data_dir'], opt_dict['checkpoint_dir'],
                    opt_dict['use_mps'])
