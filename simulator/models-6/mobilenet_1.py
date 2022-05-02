from models.model_stats import ModelStats
import numpy as np

class mobilenet_1(ModelStats):
    def __init__(
        self,
        name):
      super().__init__(name, 1)
 
      
    def update_stats(self):
        self.cpus = 6
        self.mem = 62.5
        self.speed = 62.5
        self.batch = 512
        self.placement_penalty = 1
        self.speedup = 1.0
        self.iter_time = 0.458
        self.iter_time_base = 0.71
        self.tput = np.array([
             [0.1, 0.37,0.37,0.37,0.37],
             [0.1, 0.72,0.72,0.72,0.72],
             [0.1, 1,1,1,1],
             [0.1, 1.3,1.3,1.3,1.3],
             [0.1, 1.49,1.49,1.49,1.49],
             [0.1, 1.55,1.55,1.55,1.55],
             [0.1, 1.55,1.55,1.55,1.55],
             [0.1, 1.55,1.55,1.55,1.55],
             [0.1, 1.55,1.55,1.55,1.55]
             ])

                             
      

     
      

     
