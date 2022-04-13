class GPU:
    
    def __init__(
        self,
        rack_id,
        machine_id,
        gpu_id,
        job_id,
        tenant_id,
        server_handle=None):
        # gpu details
        self.rack_id = rack_id
        self.machine_id = machine_id
        self.gpu_id = gpu_id

        # gpu state
        self.job_id = job_id
        self.tenant_id = tenant_id
        self.server_handle = server_handle

    def allocate(self, job):
        self.job_id = job.job_id
        self.tenant_id = job.tenant_id
        self.server_handle.allocated_gpus.append(self)

    def free(self):
        self.job_id = -1
        self.tenant_id = -1
        self.server_handle.allocated_gpus.remove(self)

    @property
    def server(self):
        return self.machine_id

    def is_free(self):
        return (self.tenant_id == -1)

    def __eq__(self, other):
        return (self.rack_id == other.rack_id and\
                self.machine_id == other.machine_id and\
                self.gpu_id == other.gpu_id)
    
    def __str__(self):
        return "gpu:%s:%s:%s" % (self.rack_id, self.machine_id, self.gpu_id)
