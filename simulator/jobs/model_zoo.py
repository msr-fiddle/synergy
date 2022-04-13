import os
from enum import Enum
import logging
import random

from jobs.model import Model
from jobs.task import Task, TaskName

class ModelAssignment(Enum):
    ''' Picks the next model to be assigned to the job based
    on the split of models assigned to all jobs in the cluster so far,
    or based on only the runnable jobs ( ignore finished jobs)
    RANDOM picks a model from model_zoo randomly, while FAIR does a
    round robin across available models
    '''
    OVERALL=1
    RUNNABLE=2
    RANDOM=3
    FAIR=4

class ModelZoo():
    '''
    Tracks the split of different classes of models in the workload 
    and decides the model to be assigned to the next chosen 
    job from the workload trace, in accoradance with the user-given
    model split.
    '''

    def __init__(
        self,
        image_percent=34,
        lang_percent=33,
        speech_percent=33,
        assignment = ModelAssignment.RUNNABLE):

        # logger handle 
        self.logger = logging.getLogger(__name__)

        # initialize model splits
        self.percent_image_models = image_percent
        self.percent_lang_models = lang_percent
        self.percent_speech_models = speech_percent

        # assignment option
        self.assignment = assignment

        # Percent split of models among the currently 
        # runnable jobs
        self.tasks =dict()

        self.model_id_map = dict()

        # A map of all possible models to their handle
        self.model_zoo = dict()
        self.model_zoo_multigpu = dict()
        self.next_class_idx = 0

        self.tasks = self.create_default_tasks()
        self.model_zoo = self.create_default_models()
        self.model_zoo_multigpu = self.create_models_multigpu()
        

        self.logger.info("Created a model zoo with {} models".format(len(self)))
        for key,val in self.model_zoo.items():
            self.logger.info("{} : {}, {}, {}".format(key, val.model_name, val.model_task, val.model_id))

    def __len__(self):
        return len(self.model_zoo.keys())   

    def model(self, model_id, gpu_demand):
        #if model_id in self.model_zoo:
        #    return self.model_zoo[model_id]
        if (model_id, gpu_demand) in self.model_zoo_multigpu:
            return self.model_zoo_multigpu[(model_id, gpu_demand)]
        else:
            return None

    def print_task_splits(self):
        tot = 0
        for model in self.model_zoo.values():
            self.logger.info("{}:{}".format(model.model_name, model.total_jobs))
            tot += model.total_jobs
        self.logger.info("Total jobs = {}".format(tot))

    def task(self, model_id):
        return self.model_zoo[model_id].model_task

    # based on the current or overall split, 
    # return the class id for the next job
    def get_job_class(self):

        if self.assignment == ModelAssignment.OVERALL:
            # Create a list of tasks sorted by their priorities
            # Each list entry is a tuple (task_name, task_object)
            sorted_tasks = sorted(self.tasks.items(), key=lambda x: x[1].overall_priority, reverse=True)
        elif self.assignment == ModelAssignment.RUNNABLE:
            sorted_tasks = sorted(self.tasks.items(), key=lambda x: x[1].runnable_priority, reverse=True)
        num_jobs = sum(list(task.total_jobs for task in self.tasks.values()))
        #if self.assignment == ModelAssignment.OVERALL or self.assignment == ModelAssignment.RUNNABLE:
        #    print("{}:{:.2f}:{}:{}, {}:{:.2f}:{}:{}, {}:{:.2f}:{}:{}".format(str(sorted_tasks[0][0]), sorted_tasks[0][1].overall_priority, sorted_tasks[0][1].total_jobs, sorted_tasks[0][1].runnable_jobs, str(sorted_tasks[1][0]), sorted_tasks[1][1].overall_priority, sorted_tasks[1][1].total_jobs, sorted_tasks[1][1].runnable_jobs, str(sorted_tasks[2][0]), sorted_tasks[2][1].overall_priority, sorted_tasks[2][1].total_jobs, sorted_tasks[2][1].runnable_jobs))

        num_class_id = len(self)
        # This randint fn must be called irespective of the
        # assignment type to ensure the rand generator state
        # is identical across all options
        next_class_id = random.randint(0, num_class_id-1) 
        next_task = self.task(next_class_id) 

        if self.assignment is ModelAssignment.FAIR:
            next_class_id = self.next_class_idx % num_class_id
            self.next_class_idx += 1
            next_task = self.task(next_class_id) 
        elif self.assignment is not ModelAssignment.RANDOM:
            task = sorted_tasks[0][1]
            next_task = task.task_name
            model = task.get_next_model
            next_class_id = model.model_id

        return (next_task, next_class_id)

    def get_job_class_by_name(self, name):
        if name in self.model_id_map:
            class_id = self.model_id_map[name]
            task = self.model_zoo[class_id].model_task
            return (task, class_id)
        else:
            return (TaskName.IMAGE, 0)

    # Called on  a job completion
    def remove_runnable_job(self, class_id):
        model = self.model_zoo[class_id]
        model.runnable_jobs -= 1
        self.update_task(model.model_task, -1) 

    # Called on a job arrival
    def add_runnable_job(self, class_id):
        model = self.model_zoo[class_id]
        model.total_jobs += 1
        model.runnable_jobs += 1
        self.update_task(model.model_task, 1)
        self.update_priorities() 

    def update_task(self, model_task, counter=0):
        self.tasks[model_task].runnable_jobs += counter
        if counter > 0:
            self.tasks[model_task].total_jobs += counter

    def update_priorities(self):
        for task in self.tasks.values():
            task.update_priority(self.total_jobs, self.runnable_jobs)
        #print("image ({} : {:.2f}), lang ({} : {:.2f}), speech ({} : {:.2f})".format(self.tasks[TaskName.IMAGE].total_jobs, self.tasks[TaskName.IMAGE].overall_priority, self.tasks[TaskName.LANGUAGE].total_jobs, self.tasks[TaskName.LANGUAGE].overall_priority, self.tasks[TaskName.SPEECH].total_jobs, self.tasks[TaskName.SPEECH].overall_priority))
                

    @property
    def total_jobs(self):
        return sum(list(model.total_jobs for model in self.model_zoo.values()))

    @property
    def runnable_jobs(self):
        return sum(list(model.runnable_jobs for model in self.model_zoo.values()))

    def create_default_tasks(self):
        tasks = {}
        tasks[TaskName.IMAGE] = Task(TaskName.IMAGE, self.percent_image_models)
        tasks[TaskName.LANGUAGE] = Task(TaskName.LANGUAGE, self.percent_lang_models)
        tasks[TaskName.SPEECH] = Task(TaskName.SPEECH, self.percent_speech_models)
        return tasks

    def create_models_multigpu(self):
        for class_id, model in self.model_zoo.items():
            for gpu in [1,2,4,8,16]:
               idx = (class_id, gpu)
               model = Model(model.model_name, model.model_task, class_id, gpu)
               model.use_scores_from_tput()
               #model.use_real_scores()
               self.model_zoo_multigpu[idx] = model
        return self.model_zoo_multigpu

    def create_default_models(self):
        image_options = ['alexnet', 'res18', 'res50', 'mobilenet', 'shufflenet']
        #image_options = ['alexnet', 'res18', 'res50', 'mobilenet', 'shufflenet', 'ssd']
        lang_options = ['gnmt', 'transformer', 'lstm']
        #lang_options = ['gnmt', 'bert', 'lstm', 'transformer', 'vae']
        #speech_options = []
        #speech_options = ['deepspeech']
        speech_options = ['m5', 'deepspeech']
        model_map = {}
        class_id = 0
        for model_name in image_options:
            model = Model(model_name, TaskName.IMAGE, class_id)
            model_map[class_id] = model
            self.tasks[TaskName.IMAGE].add_model(model)
            self.model_id_map[model_name]=class_id
            class_id += 1
        for model_name in lang_options:
            model = Model(model_name, TaskName.LANGUAGE, class_id)
            model_map[class_id] = model
            self.tasks[TaskName.LANGUAGE].add_model(model)
            self.model_id_map[model_name]=class_id
            class_id += 1
        for model_name in speech_options:
            model = Model(model_name, TaskName.SPEECH, class_id)
            model_map[class_id] = model
            self.tasks[TaskName.SPEECH].add_model(model)
            self.model_id_map[model_name]=class_id
            class_id += 1

        for model_id in model_map.keys():
            model = model_map[model_id]
            fname = './models/' + model.model_name
            if os.path.exists(fname):
                model.update_res_score_from_json(fname)
            else:
                model.use_scores_from_tput()
                #model.use_real_scores()
                #model.use_approx_scores()

        return model_map


               
