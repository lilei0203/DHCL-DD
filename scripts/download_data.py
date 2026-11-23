"""Download pinned public benchmark releases, preserving upstream bytes and hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'douban-book': {
        'repository': 'librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding',
        'commit': 'cc05c484b628b08237f0a876fcebf99d736ccbea',
        'files': {'user_book.dat': 'Douban Book/user_book.dat', 'UPSTREAM_README.md': 'README.md'},
        'format': 'explicit user item rating; ratings >= 4 become positives',
    },
    **{name: {
        'repository': 'kuandeng/LightGCN',
        'commit': 'b067ec05f16a24e0e1efe63cd82385f0c469af20',
        'files': {**{f: f'Data/{name}/{f}' for f in ['train.txt', 'test.txt', 'user_list.txt', 'item_list.txt']},
                  'UPSTREAM_README.md': 'README.md'},
        'format': 'binarized adjacency lists; upstream train+test union is re-split independently',
    } for name in ['yelp2018', 'amazon-book']},
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def download(name):
    spec = SOURCES[name]
    dest = ROOT / 'data' / name / 'raw'
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for local, remote in spec['files'].items():
        url = f"https://raw.githubusercontent.com/{spec['repository']}/{spec['commit']}/{urllib.parse.quote(remote)}"
        target = dest / local
        if not target.exists():
            partial = target.with_suffix(target.suffix + '.part')
            for attempt in range(4):
                try:
                    req = urllib.request.Request(url, headers={'User-Agent': 'DHCL-DD-reconstruction/1.0'})
                    with urllib.request.urlopen(req, timeout=90) as response, open(partial, 'wb') as out:
                        while block := response.read(1 << 20): out.write(block)
                    partial.replace(target)
                    break
                except Exception:
                    if attempt == 3: raise
                    time.sleep(2 ** attempt)
        records.append({'file': local, 'url': url, 'sha256': sha256(target), 'bytes': target.stat().st_size})
        print(f'{name}: {local} {target.stat().st_size:,} bytes', flush=True)
    manifest = {**{k:v for k,v in spec.items() if k != 'files'}, 'files': records,
                'status': 'public upstream benchmark copy, not recovered original DHCL-DD experiment data',
                'rights': 'Third-party data. Upstream availability does not confer a new redistribution license; retain source terms and verify rights before public archiving.'}
    (dest.parent/'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['all',*SOURCES], default='all')
    args=parser.parse_args()
    names=list(SOURCES) if args.dataset=='all' else [args.dataset]
    with ThreadPoolExecutor(max_workers=3) as pool: list(pool.map(download,names))
