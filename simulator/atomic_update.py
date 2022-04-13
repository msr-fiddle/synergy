import threading

class AtomicUpdate(object):
    
        def __init__(self):
            self._value = 0
            self._lock = threading.Lock()
    
        def inc(self, val=1):
            with self._lock:
                self._value += val
                return self._value
    
        def dec(self, val=1):
            with self._lock:
                self._value -= val
                return self._value
    
        @property
        def value(self):
            return self._value
