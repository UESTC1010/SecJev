"""Prepare pinned upstream models, code and public corpus for a reproducible run."""
import argparse, gzip, hashlib, json, os, pathlib, shutil, tarfile, urllib.request
from huggingface_hub import snapshot_download

BASE = ('Qwen/Qwen3.5-0.8B-Base', 'dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68')
INIT = ('jaredpalmer/kev-0.8b', '54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8')
KEV = '90990a5fac2995b9faa3190f7d437e84f2067768'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=pathlib.Path, required=True, help='Unpacked SecJev-Corpus v1.0.0 directory')
    args = parser.parse_args()
    root = pathlib.Path(os.environ.get('SECJEV_WORK', './secjev-work')).resolve()
    for d in ['data', 'results', 'logs', 'runs', 'source']:
        (root/d).mkdir(parents=True, exist_ok=True)
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
    for name, (repo, revision) in [('base', BASE), ('init', INIT)]:
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
    print('Prepared', root)


if __name__ == '__main__':
    main()
