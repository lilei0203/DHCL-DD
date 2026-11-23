"""Build local transport archives; this does not publish files or mint a DOI."""
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.venv','__pycache__','.pytest_cache','.git','cache'}

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def candidates():
    for folder,dirs,files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in EXCLUDED]
        for name in sorted(files):
            if name.endswith(('.pyc','.part','.tmp','.zip')) or name.startswith('~$'):continue
            yield Path(folder)/name

if __name__=='__main__':
    out=ROOT.parent/'outputs'/'DHCL-DD_delivery';out.mkdir(parents=True,exist_ok=True)
    paths=sorted(candidates())
    inventory={str(p.relative_to(ROOT)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in paths}
    (out/'file_manifest.json').write_text(json.dumps(inventory,indent=2),encoding='utf8')
    for kind in ['project','code']:
        archive=out/f'DHCL-DD_{kind}.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path in paths:
                relative=path.relative_to(ROOT)
                if kind=='code' and relative.parts[0] in ['data','runs']:continue
                z.write(path,Path('DHCL-DD')/relative)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
        print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha256(archive)}),flush=True)
