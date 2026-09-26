from paths import WORK,SCRIPTS,SOURCE,BASE
import sys, pathlib
import evaluate_retention_2b as e
e.ROOT = pathlib.Path(str(WORK / 'general'))
e.MODELS = {f'epoch-{i}': f'{WORK}/runs/secjev-2b-3epochs/epoch-{i}-calibrated' for i in [1, 2, 3]}
e.MODELS['kev-stage1'] = str(WORK / 'runs/kev-2b-delta/epoch-1')
if len(sys.argv) > 2:
    e.SUITES = {sys.argv[2]: e.SUITES[sys.argv[2]]}
e.worker(sys.argv[1])
