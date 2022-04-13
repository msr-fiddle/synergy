from models.model_stats import ModelStats
import numpy as np

class res50_1(ModelStats):
    def __init__(
        self,
        name):
      super().__init__(name, 1)
 
      
    def update_stats(self):
        self.cpus = 4
        self.mem = 62.5
        self.speed = 62.5
        self.batch = 512
        self.placement_penalty = 1
        self.speedup = 1.15
        self.iter_time = 0.78
        self.iter_time_base = 0.78
        self.tput = np.array([
             [0.1, 0.4,0.4,0.4,0.4],
             [0.1, 0.74,0.74,0.74,0.74],
             [0.1, 1,1,1,1],
             [0.1, 1.15,1.15,1.15,1.15],
             [0.1, 1.15,1.15,1.15,1.15],
             [0.1, 1.15,1.15,1.15,1.15],
             [0.1, 1.15,1.15,1.15,1.15],
             [0.1, 1.15,1.15,1.15,1.15],
             [0.1,1.15,1.15,1.15,1.15]
             ])

                             
      

     
      

     
