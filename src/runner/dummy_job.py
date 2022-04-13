from Runner.run import Runner

class DummyRunner(Runner):
	
	def __init__(self, job_name):
		self.job_name = job_name
