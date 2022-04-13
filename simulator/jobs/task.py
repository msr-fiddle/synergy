import os
from enum import Enum
import random

class TaskName(Enum):
    IMAGE=1
    LANGUAGE=2
    SPEECH=3

    def __str__(self):
        if self.value == 1:
            return "IMAGE"
        elif self.value == 2:
            return "LANG"
        elif self.value == 3:
            return "SPEECH"
        else:
            return "UNKNOWN"

''' Task characteristics
    - task_name 
'''
class Task():
    def __init__(
        self,
        task_name,
        task_percent):
        self.task_name = task_name 
        self.task_percent = task_percent
        self.total_jobs = 0
        self.runnable_jobs = 0
        self.overall_priority = 0
        self.runnable_priority = 0
        self.models = []
        self.last_model = 0

    def update_priority(self, total=1, runnable=1):
        if total > 0:
            current_share = self.total_jobs/total
            self.overall_priority = self.task_percent/100 - current_share
        if runnable > 0:
            current_share = self.runnable_jobs/runnable
            self.runnable_priority = self.task_percent/100 - current_share
       
    def add_model(self, model):
        if model is not None:
            self.models.append(model)

    @property
    def get_next_model(self):
        num_models = len(self.models)
        if num_models > 0:
            model_idx = self.last_model % num_models
            self.last_model += 1
            # get a model randomly
            #model_idx = random.randint(0, num_models-1)
            return self.models[model_idx]
        else:
            return None
            
