from enum import IntEnum

class EventType(IntEnum):

    JOB_ARRIVAL = 1
    JOB_END = 2
    JOB_LEASE_END = 3
    SCHEDULE = 4
    ALLOCATION = 5
    DEPLOY = 6

    @staticmethod
    def get_name(event_type_value):
        return EventType(event_type_value).name
