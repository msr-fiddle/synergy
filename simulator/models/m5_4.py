from models.model_stats import ModelStats 
import numpy as np

class m5_4(ModelStats):
    def __init__(
        self,
        name):
      super().__init__(name, 4)
 
      
    def update_stats(self):
        self.cpus = 6
        self.mem = 62.5*3
        self.speed = 62.5*3
        self.batch = 512
        self.placement_penalty = 1
        self.speedup = 2.5
        self.iter_time = 0.6
        self.tput = np.array([
             [0.3,0.3,0.3,0.3],
             [0.66,0.66,0.66,0.66],
             [1,1,1,1],
             [1.2,1.2,1.2,1.2],
             [1.4,1.4,1.4,1.4],
             [1.5,2,2.5,2.5],
             [1.5,2,2.5,2.5],
             [1.5,2,2.5,2.5],
             [1.5,2,2.5,2.5]
             ])

                             
             #[3.83,3.83,3.83, 3.83]
      

     
      

     
