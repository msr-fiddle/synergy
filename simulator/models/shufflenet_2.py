from models.model_stats import ModelStats
import numpy as np

class shufflenet_2(ModelStats):
    def __init__(
        self,
        name):
      super().__init__(name, 2)
 
      
    def update_stats(self):
        self.cpus = 9
        self.mem = 62.5
        self.speed = 62.5
        self.batch = 512
        self.placement_penalty = 1
        self.speedup = 2.33
        self.iter_time = 0.615
        self.tput = np.array([
             [0.3,0.3,0.3,0.3],
             [0.66,0.66,0.66,0.66],
             [1,1,1,1],
             [1.25,1.25,1.25,1.25],
             [1.53,1.53,1.53,1.53],
             [1.75,1.75,1.75,1.75],
             [2.33,2.33,2.33,2.33],
             [0,0,0,0],
             [0,0,0,0]
             ])

                             
             #[4.54,4.54,4.54, 4.54]
      

     
      

     
