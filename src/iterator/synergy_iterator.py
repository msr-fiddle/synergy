#/usr/bin/env python3

from collections.abc import Iterable
import logging
import os
import sys
import time
import traceback
import math

#from runtime.rpc import job_client

EXTEND_LEASE_FRACTION = 0.75
LOG_FORMAT = '[{asctime}] [{event}] [{status}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
except:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

class SynergyIterator:
    def __init__(self, data_loader, bs=128, dali=True,
                 steps_this_epoch=0, epoch=0, worker_id=0,
                 steps_to_run=-1,
                 load_checkpoint_func=None,
                 save_checkpoint_func=None, synthetic_data=False, mock=False):
        if not isinstance(data_loader, Iterable):
            raise ValueError("Data loader must be iterable: %s" % data_loader)
        self._data_loader = data_loader
        self._iterator = iter(self._data_loader)
        self._load_checkpoint_func = load_checkpoint_func
        self._save_checkpoint_func = save_checkpoint_func

        print("Using Synergy Iterator")
        self._steps_this_epoch = steps_this_epoch
        self._samples_this_epoch = 0
        self._epoch = epoch
        self._batch_size = bs
        self._dali = dali
        self._mock = mock
        self._steps_to_run = steps_to_run
        if not self._mock:
            self._max_steps_per_epoch = int(math.ceil(self._size / self._batch_size))
        else:
            self._max_steps_per_epoch = len(self._data_loader)
        self._total_steps =  self._epoch*self._max_steps_per_epoch + self._steps_this_epoch
        self._worker_id = worker_id
        self._steps_this_run = 0
        self._exit = False


        self._job_id = 0
        #self._job_id = int(os.environ['SYNERGY_JOB_ID'])
        #self._sched_addr = os.environ['SYNERGY_SCHED_ADDR']
        #self._sched_port = int(os.environ['SYNERGY_SCHED_PORT'])
        self._log_dir = "./logs-synergy/"
        self._log_file = os.path.join(self._log_dir, "job-%s-%s.log" %\
            (self._job_id, round(time.time())))
        self._init_logger()
        self._synthetic_data = synthetic_data
        if self._synthetic_data:
            self._initial_val = None
        self._round = 0
        self._total_time_elapsed_s = 0
        self.wait_time = 0
        self._steps_this_round = 0
        self._time_elapsed_this_round_s = 0
        self._sent_update_iters_this_round = False
        self._round_duration_s = None
        self._num_steps_per_round = None
        self._prev_time = None
        #self._rpc_client = job_client.JobRpcClient(\
        #    self._job_id, self._sched_addr, self._sched_port)
        self._write_info()


    @property
    def batch_size(self):
        return self._data_loader.batch_size

    @property
    def dataset(self):
        return self._data_loader.dataset

    def __len__(self):
        return len(self._data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        cur_time = time.time()
        # If this is the first time next is called, register with the scheduler
        if self._prev_time is None:
            print("Must register job now")
            elapsed_time = 0
            self._round_duration_s = 100000
            #elapsed_time, self._round_duration_s = self._rpc_client.register_job()
            #self._logger.info("Round duraton = %s s",self._round_duration_s, 
            #              extra={'event': 'PROGRESS', 'status': 'DURATION'})
        else:
            elapsed_time = cur_time - self._prev_time
        self._total_time_elapsed_s += elapsed_time
        self._time_elapsed_this_round_s += elapsed_time
        self._prev_time = cur_time

        # Extend the lease if necessary.
        send_update_iters = self._num_steps_per_round is not None and\
            self._steps_this_round > self._num_steps_per_round * EXTEND_LEASE_FRACTION
        send_update_iters = send_update_iters or self._time_elapsed_this_round_s >\
            self._round_duration_s * EXTEND_LEASE_FRACTION
        send_update_iters = send_update_iters and not self._sent_update_iters_this_round
        if send_update_iters:
            # TODO: check remaining steps here
            extend_lease = True
            #self._num_steps_per_round = self._rpc_client.update_iters(\
            #    self._steps_this_round, self._time_elapsed_this_round_s - self.wait_time, extend_lease, self._round)
            self._sent_update_iters_this_round = True
            self._logger.info("Lease extended, ran {} steps for {:.2f} seconds so far".format(
                self._total_steps, self._total_time_elapsed_s),\
                extra={'event': 'LEASE', 'status': 'EXTEND'})

        # Notify scheduler that the round has ended for this job
        # Note: for round end, we only check num steps instead of time elapsed, because
        # we want to ensure all workers of the same job runs exactly the same number of
        # iterations per round, so as to avoid hanging.
        time_before_sync = time.time()
        if self._num_steps_per_round is not None and\
                self._steps_this_round >= self._num_steps_per_round:
            lease_expired = self.send_round_end()
            time_after_sync = time.time()
            self.wait_time = time_after_sync - time_before_sync
            
            if lease_expired:
                self._exit = True
                #TODO  : self.complete()
                if not self._mock:
                    import torch
                    if torch.distributed.is_initialized():
                        torch.distributed.barrier()
                raise StopIteration

        # Return a new data item if one exists.
        try:
            if self._synthetic_data and self._initial_val is not None:
                val = self._initial_val
            else:
                val = next(self._iterator)
                if self._synthetic_data and self._initial_val is None:
                    self._initial_val = val
            self._total_steps += 1
            self._steps_this_round += 1
        except StopIteration as e:
            self._write_info()
            raise e

        if self._synthetic_data and self._total_steps % len(self._data_loader) == 0:
            raise StopIteration
        #TODO : remove this
        #time.sleep(0.05)
        return val

    def send_round_end(self):
        """
        Notify the scheduler that the current round has ended for this job.
        Return whether the scheduler terminated our lease.
        """
        #lease_expired = self._rpc_client.round_end(\
        #    self._steps_this_round, self._time_elapsed_this_round_s, self._exit)
        if self._mock:
            print("* Round end: lease expired = %s" % lease_expired)
        self._round += 1
        self._steps_this_round = 0
        self._time_elapsed_this_round_s = 0
        self._sent_update_iters_this_round = False
        if lease_expired:
            self._exit = True
            self._logger.info("Lease expired, ran {} steps for {:.2f} seconds".format(
                self._total_steps, self._total_time_elapsed_s),\
                extra={'event': 'LEASE', 'status': 'EXPIRED'})
        return lease_expired

    @property
    def exit(self):
        return self._exit
    @property
    def sampler(self):
        return self._data_loader.sampler

    @property
    def _size(self):
        if hasattr(self._data_loader, '_size'):
            return self._data_loader._size
        else:
            return len(self._data_loader) 
            #try:
            #  length = len(self._data_loader)
            #except:
            #  length = 0
            #return length

    """
    Send lease end notification to scheduler and clean up
    Should be called by job script when training completes, or by the iterator 
    if lease has been terminated by the scheduler
    """
    def complete(self):
        self._exit = True
        #if not self._sent_update_lease_this_round and self._steps_this_round > 0:
        #    self._rpc_client.lease_ended(self._steps_this_round, self._time_elapsed_this_round_s)   
        #self._rpc_client.lease_ended(self._steps_this_round, self._time_elapsed_this_round_s, self._round)
        self._write_info()
        self._logger.info('', extra={'event': 'LEASE', 'status': 'COMPLETE'})
        self._logger.removeHandler(self._file_handler)
        self._file_handler.close()

    def load_checkpoint(self, *args, **kwargs):
        self._logger.info('', extra={'event': 'LOAD CHECKPOINT',
                                     'status': 'BEGIN'})
        checkpoint = self._load_checkpoint_func(*args, **kwargs)
        self._logger.info('', extra={'event': 'LOAD CHECKPOINT',
                                     'status': 'END'})
        return checkpoint

    def save_checkpoint(self, *args, **kwargs):
        self._logger.info('', extra={'event': 'SAVE CHECKPOINT',
                                     'status': 'BEGIN'})
        retval = self._save_checkpoint_func(*args, **kwargs)
        self._logger.info('', extra={'event': 'SAVE CHECKPOINT',
                                     'status': 'END'})
        return retval

    def _init_logger(self):
        self._logger = logging.getLogger('synergy_iterator')
        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)
        self._logger.propagate = False
        self._logger.setLevel(logging.DEBUG)
        self._file_handler = logging.FileHandler(self._log_file)
        self._file_handler.setFormatter(
                logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT, style='{'))
        self._file_handler.setLevel(logging.DEBUG)
        self._logger.addHandler(self._file_handler)

    def _write_info(self):
        self._logger.info('{0}'.format(self._total_steps),
                          extra={'event': 'PROGRESS', 'status': 'STEPS'})
        self._logger.info('{0}'.format(self._total_time_elapsed_s),
                          extra={'event': 'PROGRESS', 'status': 'DURATION'})

# For testing purposes only
if __name__ == "__main__":
    args = sys.argv
    if len(args) != 3:
        print("Usage: ./synergy_iterator.py [sched_addr] [sched_port]")
        sys.exit(1)
    sched_addr, sched_port = args[1], args[2]
    data_loader = list(range(50))
    os.environ["SYNERGY_JOB_ID"] = "123"
    os.environ["SYNERGY_WORKER_ID"] = "234"
    os.environ["SYNERGY_SCHED_ADDR"] = sched_addr
    os.environ["SYNERGY_SCHED_PORT"] = sched_port
    os.environ["SYNERGY_LOG_DIR"] = "."
    os.environ["SYNERGY_DEBUG"] = "true"
    print("Registering job with scheduler at %s:%s" % (sched_addr, sched_port))
    synergy_iterator = SynergyIterator(data_loader, mock=True)
    print("Created Synergy iterator, waiting a few seconds before starting...")
    time.sleep(3)
    for i, _ in enumerate(synergy_iterator):
        print("Processing iteration {}/{} : R {} : Steps : {}/{}".format(i+1, len(data_loader), synergy_iterator._round, synergy_iterator._steps_this_round, synergy_iterator._num_steps_per_round))
        time.sleep(0.5)
    synergy_iterator.complete()
    print("All done!")

