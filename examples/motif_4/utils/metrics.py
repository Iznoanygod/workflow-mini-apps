import time, sys

class Timer:
    def __init__(self): self.t0 = None
    def start(self): self.t0 = time.time(); return self
    def stop(self): return time.time() - self.t0

def log(msg): print(f"[dt] {msg}", file=sys.stdout, flush=True)