from queue import PriorityQueue

class EventQueue:

    def __init__(self):
        self.event_queue = PriorityQueue()

    def empty(self):
        return self.event_queue.empty()

    def put(self, event):
        self.event_queue.put(event)
    
    def get(self):
        return self.event_queue.get()

    def __str__(self):
        s = '{ \n'
        for event in self.event_queue.queue:
            s += '\t' + str(event) + '\n'
        s += '} \n'
        return s
