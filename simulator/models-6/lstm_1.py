from models.model_stats import ModelStats    
import numpy as np

class lstm_1(ModelStats):
    def __init__(
        self,
        name):
      super().__init__(name, 1)
 
      
    def update_stats(self):
        self.cpus = 1
        self.mem = int(62.5/3)
        self.speed = int(62.5/3)
        self.batch = 20
        self.placement_penalty = 1
        self.speedup = 1
        self.iter_time = 0.011
        self.iter_time_base = 0.011
        self.tput = np.array([
             [1,1,1,1,1],
             [1, 1,1,1,1],
             [1, 1,1,1,1],
             [1, 1,1,1,1],
             [1, 1,1,1,1],
             [1, 1,1,1,1],
             [1, 1,1,1,1],
             [1, 1,1,1,1],
             [1, 1,1,1,1]
             ])

                             
      

     
      

     
