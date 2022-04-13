import logging
from events.event_type import EventType
import time

class ClusterEvent:

    runner = None

    def __init__(self, time, event_type):
        self.logger = logging.getLogger(__name__)
        self.time = time
        self.event_type = event_type

    def handleEvent(self):
        self.logger.debug(self)
        ClusterEvent.runner.set_time(self.time)

    def __lt__(self, other):
        if self.time == other.time:
            return self.event_type < other.event_type
        else:
            return self.time < other.time

    def __eq__(self, other):
        return self.time == other.time and \
            self.event_type == other.event_type

    def __str__(self):
     return "event:{:.2f}:{:.2f}:{}".format(self.time, time.time() - ClusterEvent.runner.real_start_time, EventType.get_name(self.event_type))
