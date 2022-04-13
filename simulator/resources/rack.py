import logging
import numpy as np
import os
from resources.server import Server
from resources.server_config import DefaultServerConfig

'''
Creates a rack with specified number of homogenous servers
If number of servers, and server config is unspecified, 
a default number of 4 standard servers are created.
'''

class Rack():

    def __init__(self, 
                 rack_id, 
                 num_servers,
                 start_server_id=None,
                 start_gpu_id=None,
                 server_config=None,
                 conn_list=None):
        self.logger = logging.getLogger(__name__)
        self.rack_id = rack_id
        self.start_server_id = start_server_id
        self.start_gpu_id = start_gpu_id
        self.num_servers = num_servers
        self.server_config = server_config
        self.conn_list = conn_list
        self.servers = []
        self.gpus = []


        if self.server_config is not None:
            if not isinstance(self.server_config, DefaultServerConfig):
                self.servers = self.create_servers() 
            else:
                self.servers = self.create_servers(custom=True)
        else: 
            self.servers = self.create_servers()

        for server in self.servers:
            for _gpu in server.gpu_list:
                self.gpus.append(_gpu)


    '''
    Creates a rack with 'num_servers' default servers
    If num_servers is not specified, a deafult of 4 servers is created
    Default server configs are as defined in server.py
    '''
    def create_servers(self, custom=False):
 
         simulate = True
         if self.conn_list is not None:
             simulate = False
             if os.path.exists(self.conn_list):
                 ip_port_list = open(self.conn_list, 'r').read().splitlines()
                 ip_port_list = [(item.split(',')) for item in ip_port_list] 
                 if len(ip_port_list) != self.num_servers:
                     raise ValueError("Mismatch in server IP list") 


         if self.start_server_id is None:
             self.start_server_id = 0

         if self.start_gpu_id is None:
             self.start_gpu_id = 0

         # Global ID of 1st GPU in the server
         gpu_id = self.start_gpu_id

         servers = []

         for i in range(self.num_servers):
             if simulate:
                 if not custom:
                     server = Server(server_id=i+self.start_server_id, start_gpu=gpu_id)
                 else:
                     server = Server(server_id=i+self.start_server_id, 
                       server_config=self.server_config, start_gpu=gpu_id)
             else:
                 if not custom:
                     server = Server(server_id=i+self.start_server_id, \
                                start_gpu=gpu_id, 
                                ip=ip_port_list[i][0], 
                                port=ip_port_list[i][1], 
                                start_gpu_deploy=ip_port_list[i][2], 
                                start_cpu_deploy=ip_port_list[i][3],
                                numa_aware=ip_port_list[i][4])
                 else:
                     server = Server(server_id=i+self.start_server_id, \
                                server_config=self.server_config, \
                                start_gpu=gpu_id, \
                                ip=ip_port_list[i][0], \
                                port=ip_port_list[i][1], \
                                start_gpu_deploy=ip_port_list[i][2], \
                                start_cpu_deploy=ip_port_list[i][3], \
                                numa_aware=ip_port_list[i][4])



             servers.append(server)
             gpu_id = server.num_gpus + gpu_id
             #self.logger.debug("Created server %s with %s GPUs," \
             #     "%s CPUs, %s GB memory, %s MB/s storage, %s Gbps Ethernet", 
             #     server.id,
             #     server.gpu_available, server.cpu_available, server.mem_available, server.sspeed_available, server.net_available)

         return servers

    @property
    def server_list(self):
        return self.servers

    @property
    def gpu_list(self):
        return self.gpus

    def print_rack_stats(self):
         for server in self.servers:
             server.print_server_stats()
         
        
    def __getstate__(self):
        d = self.__dict__.copy()
        if 'logger' in d:
            d['logger'] = d['logger'].name
        return d

    def __setstate__(self, d):
        if 'logger' in d:
            d['logger'] = logging.getLogger(d['logger'])
        self.__dict__.update(d) 
