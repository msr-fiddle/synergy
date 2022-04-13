import csv
from datetime import datetime
import json
import os
import pickle
import psutil
import random
import re
import socket
import subprocess

def get_self_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip_address = s.getsockname()[0]
    s.close()
    #hostname = socket.gethostname()
    #ip_address = socket.gethostbyname(hostname)
    return ip_address

def get_num_gpus():
    command = 'nvidia-smi -L'
    output = subprocess.run(command, stdout=subprocess.PIPE, check=True,
                            shell=True).stdout.decode('utf-8').strip()
    return len(output.split('\n'))

def get_pid_for_job(command):
    pids = []
    for proc in psutil.process_iter():
        cmdline = ' '.join(proc.cmdline())
        if cmdline == command:
            pids.append(proc.pid)
    return min(pids)

def list_as_string(list_of_int):
    str_list = [str(int_val) for int_val in list_of_int]
    return str_list 
