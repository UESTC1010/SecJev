"""Prepare pinned upstream models, code and public corpus for a reproducible run."""
import argparse, gzip, hashlib, json, os, pathlib, shutil, tarfile, urllib.request
from huggingface_hub import snapshot_download

BASE = ('Qwen/Qwen3.5-2B-Base', 'b1485b2fa6dfa1287294f269f5fb618e03d52d7c')
KEV = '90990a5fac2995b9faa3190f7d437e84f2067768'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=pathlib.Path, help='Unpacked SecJev-Corpus v1.0.0 directory')
    parser.add_argument('--inference-only', action='store_true', help='Download pinned backbone and Kev code without training data or parent adapter')
    args = parser.parse_args()
    if not args.inference_only and args.corpus is None: parser.error('--corpus is required for training setup')
    root = pathlib.Path(os.environ.get('SECJEV_WORK', './secjev-2b-work')).resolve()
    for d in ['data', 'results', 'logs', 'runs', 'source']:
        (root/d).mkdir(parents=True, exist_ok=True)
    if not args.inference_only:
        corpus = json.loads((args.corpus/'manifest.json').read_text())
        assert corpus['name'] == 'SecJev-Corpus' and corpus['version'] == '1.0.0'
        manifest = {}
        for split, info in corpus['subsets'].items():
            src = args.corpus/'records'/f'{split}.jsonl.gz'
            dest = root/'data'/src.name
            digest = hashlib.sha256()
            with gzip.open(src, 'rt') as stream:
                for line in stream:
                    r = json.loads(line)
                    payload = json.dumps({'state': r['state'], 'questions': r['questions']}, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
                    digest.update((payload+'\n').encode())
            assert digest.hexdigest() == info['content_sha256'], split
            shutil.copy2(src, dest)
            manifest[split] = {'records': info['scenes'], 'questions': info['questions'], 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
        (root/'data/manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for name, (repo, revision) in [('base', BASE)]:
        if args.inference_only and name == 'init': continue
        snapshot_download(repo, revision=revision, local_dir=root/name,
                          allow_patterns=['*.json', '*.safetensors', '*.pt', '*.txt', '*.jinja'])
        filename = 'download_provenance.json' if name == 'base' else 'inference-provenance.json'
        (root/name/filename).write_text(json.dumps({'repo_id': repo, 'revision': revision}, indent=2)+'\n')
    archive = root/'source'/f'kev-{KEV}.tar.gz'
    urllib.request.urlretrieve(f'https://codeload.github.com/jaredpalmer/kev/tar.gz/{KEV}', archive)
    with tarfile.open(archive) as tf:
        for member in tf:
            if member.isfile():
                rel = pathlib.PurePosixPath(member.name)
                assert rel.parts[0] == f'kev-{KEV}' and '..' not in rel.parts
                target = root/'source'/'kev'/pathlib.Path(*rel.parts[1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(member) as source, target.open('wb') as destination:
                    shutil.copyfileobj(source, destination)
    if not args.inference_only:
        import sys
        sys.path.insert(0,str(root/'source/kev'))
        from kev.suite import load_split, validate_training, read_manifest
        suite=root/'evals/v7/decision-v7'
        shutil.copytree(root/'source/kev/evals/v7/decision-v7',suite,dirs_exist_ok=True)
        rows=load_split(suite,'train');m=read_manifest(suite);validate_training(rows,m);assert len(rows)==12576
        # Obtain the exact public delta used in the measured two-stage run.
        night=root/'upstream/evals/night2';night.mkdir(parents=True,exist_ok=True)
        for filename in ['manifest.json','dates_unknowable.jsonl']:
            urllib.request.urlretrieve('https://raw.githubusercontent.com/jaredpalmer/kev/30c619b0527501cfdd448cb6eb9887e2af454603/evals/night2/'+filename,night/filename)
        manifest=json.loads((night/'manifest.json').read_text())
        assert hashlib.sha256((night/'dates_unknowable.jsonl').read_bytes()).hexdigest()==manifest['files']['dates_unknowable.jsonl']['sha256']=='afd8502d162163605ac446439e32c7b9083302dd78a6bfdc5e30075619e98437'
    print('Prepared', root)


if __name__ == '__main__':
    main()
