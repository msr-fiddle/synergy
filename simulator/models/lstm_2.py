from models.model_stats import ModelStats    
import numpy as np

class lstm_2(ModelStats):
    def __init__(
        self,
        name):
      super().__init__(name, 2)
 
      
    def update_stats(self):
        self.cpus = 1
        self.mem = 62.5
        self.speed = 62.5
        self.batch = 20
        self.placement_penalty = 1
        self.speedup = 1
        self.iter_time = 0.011
        self.tput = np.array([
             [1,1,1,1],
             [1,1,1,1],
             [1,1,1,1],
             [1,1,1,1],
             [1,1,1,1],
             [1,1,1,1],
             [1,1,1,1],
             [0,0,0,0],
             [0,0,0,0]
             ])

                             
      

     
      

     
