from models.model_stats import ModelStats
import numpy as np

class res18_1(ModelStats):
    def __init__(
        self,
        name):
      super().__init__(name, 1)
 
      
    def update_stats(self):
        self.cpus = 9
        self.mem = 62.5
        self.speed = 62.5
        self.batch = 512
        self.placement_penalty = 1
        self.speedup = 1.568
        self.iter_time = 0.41
        self.iter_time_base = 0.64
        self.tput = np.array([
             [0.1, 0.37,0.37,0.37,0.37],
             [0.1, 0.72,0.72,0.72,0.72],
             [0.1, 1,1,1,1],
             [0.1, 1.3,1.3,1.3,1.3],
             [0.1, 1.53,1.53,1.53,1.53],
             [0.1, 1.75,1.75,1.75,1.75],
             [0.1, 2.4,2.4,2.4,2.4],
             [0.1, 2.4,2.4,2.4,2.4],
             [0.1, 2.4,2.4,2.4,2.4]
             ])

             #[2.6,2.6,2.6,2.6],
             #[3,3,3, 3]
                             
      

     
      

     
