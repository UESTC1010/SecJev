"""Continue to three epochs, select on full development, then calibrate and test."""
import pathlib,os,sys,json,time,subprocess,shutil,hashlib,traceback,pickle
P=pathlib.Path(os.environ.get('SECJEV_WORK', './secjev-work')).resolve();R=P/'continuation';RUN=R/'runs/secjev-0.8b-3epochs';SCRIPTS=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from evaluate import metrics,fit,write
from kev.checkpoint import Checkpoint,write_meta
import torch
torch.set_num_threads(4)


def stage(name,**kw):write(R/'results/pipeline-status.json',{'stage':name,'updated_unix':time.time(),**kw})
def call(name,args,env=None):
    with (R/'logs'/f'{name}.log').open('w') as f:rc=subprocess.call(args,stdout=f,stderr=subprocess.STDOUT,env=env)
    (R/'logs'/f'{name}.exit').write_text(str(rc)+'\n');assert rc==0,(name,rc)
def load_rows(path):
    with path.open() as f:return [json.loads(line) for line in f]
def validate_logits(rows,split):
    expected=pickle.load((R/'data'/f'{split}.pkl').open('rb'))
    lookup={(x['case'],x['qid']):x for x in expected}
    assert len(rows)==len(lookup)==len({(x['case'],x['qid']) for x in rows})
    for x in rows:
        row=lookup[x['case'],x['qid']]
        assert all(x[k]==row[k] for k in ['source','task','qtype','label','semantic_label','option_keys'])
        assert len(x['logits'])==row['k']
    return sorted(rows,key=lambda x:(x['case'],x['qid']))
def inference(checkpoint,tag,split):
    jobs=[]
    for gpu in range(3):
        name=f'{tag}-{split}-{gpu}';f=(R/'logs'/f'{name}.log').open('w')
        args=[sys.executable,str(SCRIPTS/'infer_shard.py'),'--checkpoint',str(checkpoint),'--tag',tag,'--split',split,'--shard',str(gpu)]
        jobs.append((name,subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2').split(',')[gpu])),f))
    errors=[]
    for name,p,f in jobs:
        rc=p.wait();f.close();(R/'logs'/f'{name}.exit').write_text(str(rc)+'\n')
        if rc:errors.append([name,rc])
    assert not errors,errors
    rows=[]
    for gpu in range(3):rows.extend(load_rows(R/'results'/f'{tag}-{split}-shard-{gpu}.jsonl'))
    rows=validate_logits(rows,split)
    with (R/'results'/f'{tag}-{split}-logits.jsonl').open('w') as f:
        for x in rows:f.write(json.dumps(x)+'\n')
    return rows
def report(rows,temp):return {'raw':metrics(rows,1.),'recalibrated':metrics(rows,temp)}
def file_hashes(folder):return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}


def main():
    assert not RUN.exists()
    # Run after initial training and its calibration/development evaluation finish.
    for name in ['parent','trained']:
        assert (P/'results'/f'{name}-validation.json').exists(), 'Run initial validation first'
    assert not (P/'data/test.pkl').exists(), 'Start from a fresh work directory; test opens after selection'
    for name in ['results','logs','runs']:
        (R/name).mkdir(parents=True,exist_ok=True)
    if not (R/'data').exists(): (R/'data').symlink_to(P/'data',target_is_directory=True)
    shutil.copy2(P/'results/data-profile.json',R/'results/data-profile.json')
    allocated=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2').split(',')
    assert len(allocated)==3, 'Set CUDA_VISIBLE_DEVICES to exactly three available GPUs'
    call('batching-check',[sys.executable,str(SCRIPTS/'check_batching.py')])
    stage('training',total_epochs=3,additional_epochs=2)
    call('training',['torchrun','--standalone','--nproc_per_node=3',str(SCRIPTS/'train_continue.py')])
    stage('development_checkpoint_selection')
    paths={1:P/'runs/secjev-0.8b-v1/checkpoint',2:RUN/'epoch-2',3:RUN/'epoch-3'}
    all_rows={1:validate_logits(load_rows(P/'results/trained-development-logits.jsonl'),'development')}
    for epoch in [2,3]:all_rows[epoch]=inference(paths[epoch],f'epoch-{epoch}','development')
    candidates=[]
    for epoch,rows in all_rows.items():
        score=metrics(rows,1.);score['macro_task_nll']=sum(x['nll'] for x in score['tasks'].values())/len(score['tasks'])
        candidates.append({'epoch':epoch,'metrics':score,'checkpoint_files':file_hashes(paths[epoch])})
    best=min(candidates,key=lambda x:(-x['metrics']['macro_task_balanced_accuracy'],x['metrics']['macro_task_nll'],x['epoch']))
    selection={'selected_epoch':best['epoch'],'rule':'maximum full-development macro-task semantic-class balanced accuracy; tie: minimum macro-task raw NLL; then earlier epoch','development_questions':len(all_rows[1]),'candidates':candidates,'test_used_for_selection':False,'selected_unix':time.time()}
    write(R/'results/checkpoint-selection.json',selection)
    config=json.loads((RUN/'training_config.json').read_text());config['selected_epoch']=best['epoch'];write(RUN/'training_config.json',config)
    shutil.copytree(paths[best['epoch']],RUN/'checkpoint',ignore=shutil.ignore_patterns('optimizer.pt'))
    selected_rows=all_rows[best['epoch']]
    with (R/'results/trained-development-logits.jsonl').open('w') as f:
        for row in selected_rows:f.write(json.dumps(row)+'\n')
    stage('selected_checkpoint_calibration',selected_epoch=best['epoch'])
    if best['epoch']==1:
        calibration=validate_logits(load_rows(P/'results/trained-calibration-logits.jsonl'),'calibration')
        shutil.copy2(P/'results/trained-calibration-logits.jsonl',R/'results/trained-calibration-logits.jsonl')
    else:calibration=inference(RUN/'checkpoint','trained','calibration')
    import torch
    with torch.enable_grad():temperature=fit(calibration)
    shutil.copytree(RUN/'checkpoint',RUN/'calibrated');ck=Checkpoint(RUN/'calibrated');ck.meta.temperature=temperature
    ck.meta.extra['temperature_fit']={'split':'calibration','questions':len(calibration),'data_sha256':json.loads((R/'data/manifest.json').read_text())['calibration']['sha256']}
    ck.meta.extra['checkpoint_selection']={'selected_epoch':best['epoch'],'split':'development','rule':selection['rule']}
    write_meta(RUN/'calibrated',ck.meta)
    write(R/'results/trained-validation.json',{'model':'trained','selected_epoch':best['epoch'],'temperature':temperature,'temperature_fitted_only_on':'calibration','calibration':report(calibration,temperature),'development':report(selected_rows,temperature),'test_used_for_training_or_selection':False})
    shutil.copy2(P/'results/parent-validation.json',R/'results/parent-validation.json')
    frozen=file_hashes(RUN/'calibrated')
    write(R/'results/frozen-final-candidate.json',{'selection_rule':selection['rule'],'selected_epoch':best['epoch'],'checkpoint':'runs/secjev-0.8b-3epochs/calibrated','files':frozen,'temperature_source':'calibration only','test_opened':False,'frozen_unix':time.time()})
    stage('final_test',selected_epoch=best['epoch'])
    call('prepare-test',[sys.executable,str(SCRIPTS/'prepare.py'),'--allow-test'])
    for name,path,temp in [('parent',str(P/'init'),json.loads((R/'results/parent-validation.json').read_text())['temperature']),('trained',RUN/'calibrated',temperature)]:
        rows=inference(path,name,'test')
        write(R/'results'/f'{name}-test.json',{'model':name,'temperature':temp,'temperature_fitted_only_on':'calibration','test':report(rows,temp),'test_used_for_training_or_selection':False})
    assert file_hashes(RUN/'calibrated')==frozen
    stage('evaluation_complete',selected_epoch=best['epoch'],checkpoint='runs/secjev-0.8b-3epochs/calibrated')
    call('finish-release',[sys.executable,str(SCRIPTS/'finish_release.py')])


if __name__=='__main__':
    try:main()
    except Exception as error:
        (R/'results').mkdir(parents=True,exist_ok=True)
        stage('failed',error=str(error),traceback=traceback.format_exc());raise
