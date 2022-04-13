from models.model_stats import ModelStats
import numpy as np

class res50_4(ModelStats):
    def __init__(
        self,
        name):
      super().__init__(name, 4)
 
      
    def update_stats(self):
        self.cpus = 3
        self.mem = 62.5
        self.speed = 62.5
        self.batch = 512
        self.placement_penalty = 1
        self.speedup = 1
        self.iter_time = 0.78
        self.tput = np.array([
             [0.4,0.4,0.4,0.4],
             [0.74,0.74,0.74,0.74],
             [1,1,1,1],
             [1,1,1,1],
             [1,1,1,1],
             [1,1,1,1],
             [0,0,0,0],
             [0,0,0,0],
             [0,0,0,0]
             ])

                             
      

     
      

     
