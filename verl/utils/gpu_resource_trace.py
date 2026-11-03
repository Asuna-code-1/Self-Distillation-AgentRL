"""Opt-in, read-only NVML tracing of native trainer phases (no CUDA allocations)."""
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import threading
import time

_INSTANCE = None


class GPUResourceTrace:
    def __init__(self, directory):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.indices = [int(x) for x in os.environ.get('RESKILL_GPU_INDICES', '0,1,2,3').split(',')]
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.indices]
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.stack = []
        self.peaks = {}
        self.step = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _write(self, name, record):
        with (self.root / name).open('a') as handle:
            handle.write(json.dumps(record) + '\n')

    def sample(self):
        used = [self.nvml.nvmlDeviceGetMemoryInfo(h).used / 2**30 for h in self.handles]
        with self.lock:
            phase = self.stack[-1] if self.stack else 'between_phases'
            peak = self.peaks.setdefault(phase, [0.] * len(used))
            self.peaks[phase] = [max(a,b) for a,b in zip(peak,used)]
            self._write('gpu_samples.jsonl', {'time':time.time(), 'step':self.step,
                        'phase':phase,'indices':self.indices,'used_gib':used})

    def _loop(self):
        while not self.stop.wait(.2):
            self.sample()

    @contextmanager
    def phase(self, name):
        with self.lock:
            self.stack.append(name)
            self._write('phases.jsonl', {'time':time.time(),'step':self.step,'phase':name,'event':'begin'})
        self.sample()
        try:
            yield
        finally:
            self.sample()
            with self.lock:
                self._write('phases.jsonl', {'time':time.time(),'step':self.step,'phase':name,'event':'end'})
                self.stack.pop()

    def metrics(self):
        with self.lock:
            result = {f'resources/{phase}_gpu_{index}_peak_gib':value
                      for phase,values in self.peaks.items() for index,value in zip(self.indices,values)}
            self._write('step_peaks.jsonl', {'time':time.time(),'step':self.step,'metrics':result})
            self.peaks = {}
            return result


def get_trace():
    global _INSTANCE
    directory = os.environ.get('RESKILL_RESOURCE_TRACE')
    if directory and _INSTANCE is None:
        _INSTANCE = GPUResourceTrace(directory)
    return _INSTANCE


def phase(name):
    @contextmanager
    def scope():
        trace = get_trace()
        if trace is None:
            yield
        else:
            with trace.phase(name):
                yield
    return scope()


def traced(name):
    def decorate(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            with phase(name):
                return function(*args, **kwargs)
        return wrapper
    return decorate
