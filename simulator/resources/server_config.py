import logging

class DefaultServerConfig():

    def __init__(self):
        self.num_cpus = 24 #CPU cores
        self.memory = 500  #in GB
        self.storage_speed = 300  #in MBps  
        self.num_gpus = 8
        self.network = 40 #Gbps


    @property
    def cpu(self):
        return self.num_cpus
        
    @property
    def mem(self):
        return self.memory

    @property
    def sspeed(self):
        return self.storage_speed

    @property
    def gpu(self):
        return self.num_gpus

    @property
    def net(self):
        return self.network

class CustomServerConfig(DefaultServerConfig): 
    
    def __init__(self, num_cpus=None, memory=None, 
                   storage_speed=None, num_gpus=None, network=None):
        super(CustomServerConfig, self).__init__()
        if num_cpus is not None:
            self.num_cpus = num_cpus
        if memory is not None:
            self.memory = memory
        if storage_speed is not None:
            self.storage_speed = storage_speed
        if num_gpus is not None:
            self.num_gpus = num_gpus
        if network is not None:
            self.network = network

        
