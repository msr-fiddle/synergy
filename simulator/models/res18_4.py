from models.model_stats import ModelStats
import numpy as np

class res18_4(ModelStats):
    def __init__(
        self,
        name):
      super().__init__(name, 4)
 
      
    def update_stats(self):
        self.cpus = 6
        self.mem = 62.5
        self.speed = 62.5
        self.batch = 512
        self.placement_penalty = 1
        self.speedup = 1.75
        self.iter_time = 0.64
        self.tput = np.array([
             [0.37,0.37,0.37,0.37],
             [0.72,0.72,0.72,0.72],
             [1,1,1,1],
             [1.3,1.3,1.3,1.3],
             [1.53,1.53,1.53,1.53],
             [1.75,1.75,1.75,1.75],
             [0,0,0,0],
             [0,0,0,0],
             [0,0,0,0]
             ])

             #[2.6,2.6,2.6,2.6],
             #[3,3,3, 3]
                             
      

     
      

     
