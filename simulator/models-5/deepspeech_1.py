from models.model_stats import ModelStats
import numpy as np

class deepspeech_1(ModelStats):
    def __init__(
        self,
        name):
      super().__init__(name, 1)
 
      
    def update_stats(self):
        self.cpus = 3
        self.mem = 62.5
        self.speed = 62.5
        self.batch = 20
        self.placement_penalty = 1
        self.speedup = 1
        self.iter_time = 0.757
        self.iter_time_base = 0.757
        self.tput = np.array([
             [0.1, 0.3,0.3,0.3,0.3],
             [0.1, 0.66,0.66,0.66,0.66],
             [0.1, 1,1,1,1],
             [0.1, 1,1,1,1],
             [0.1, 1,1,1,1],
             [0.1, 1,1,1,1],
             [0.1, 1,1,1,1],
             [0.1, 1,1,1,1],
             [0.1, 1,1,1,1]
             ])

                             
      

     
      

     
