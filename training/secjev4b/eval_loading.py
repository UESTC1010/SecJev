from paths import WORK,SCRIPTS,SOURCE,BASE
"""Limit transient host memory when loading nine FP32 replicas."""
import fcntl, gc

def load_serial(checkpoint, device, options):
    with open(str(WORK / 'evaluation-model-load.lock'), 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = checkpoint.load(device, options)
        gc.collect()
        return result
