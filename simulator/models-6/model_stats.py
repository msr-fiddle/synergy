class ModelStats:
    def __init__(
        self,
        name,
        gpus):
      self.name = name
      self.gpus = gpus
      self.tput = None
      self.cpus = 3
      self.mem = 62.5
      self.speed = 62.5
      self.batch = 512
      self.placement_penalty = 1
      self.iter_time = 1
      self.speedup = 1
      self.update_stats()
    
    def update_stats(self):
        pass

    def __str__(self):
        return "{}:{}:{}".format(self.name, self.cpus, self.iter_time)
     
