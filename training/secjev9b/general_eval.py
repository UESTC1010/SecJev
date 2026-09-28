from paths import WORK,SCRIPTS,SOURCE,BASE
import sys, pathlib
import evaluate_retention_9b as e
e.ROOT = pathlib.Path(str(WORK / 'general'))
e.MODELS = {f'epoch-{i}': f'{WORK}/runs/secjev-9b-3epochs/epoch-{i}-calibrated' for i in [1, 2, 3]}
e.MODELS['kev-9b'] = str(WORK / 'init')
if len(sys.argv) > 2:
    e.SUITES = {sys.argv[2]: e.SUITES[sys.argv[2]]}
e.worker(sys.argv[1])
