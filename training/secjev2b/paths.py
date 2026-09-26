import os,sys,pathlib
WORK=pathlib.Path(os.environ.get('SECJEV_WORK','./secjev-2b-work')).resolve()
SCRIPTS=pathlib.Path(__file__).resolve().parent
SOURCE=WORK/'source/kev'
BASE=str(WORK/'base')
sys.path.insert(0,str(SOURCE))
