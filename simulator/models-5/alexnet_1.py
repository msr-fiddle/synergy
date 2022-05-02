from models.model_stats import ModelStats 
import numpy as np

class alexnet_1(ModelStats):
    def __init__(
        self,
        name):
      super().__init__(name, 1)
 
      
    def update_stats(self):
        self.cpus = 12
        self.mem = 62.5*2
        self.speed = 62.5*2
        self.batch = 512
        self.placement_penalty = 1
        self.speedup = 1.92
        self.iter_time = 0.392
        self.iter_time_base = 0.6
		#Still a 3-CPu tput matrix. Divide by 1.25
        self.tput = np.array([
             [0.1,0.3,0.3,0.3,0.3],
             [0.1,0.66,0.66,0.66,0.66],
             [0.1,1,1,1,1],
             [0.1,1.25,1.25,1.25,1.25],
             [0.1,1.53, 1.53,1.53,1.53],
             [0.1,1.75, 1.75,1.75,1.75],
             [0.1,2, 2,2,2],
             [0.1,2,2.94,2.94,2.94],
             [0.1,2,2.94,2.94,2.94]
             ])

                             
             #[3.83,3.83,3.83, 3.83]
      

     
      

     
