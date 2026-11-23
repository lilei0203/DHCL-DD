"""Audit hashes, split disjointness and train-support coverage for all datasets."""
from pathlib import Path
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dhcldd.data import Dataset,digest

def check(name):
    folder=ROOT/'data'/name
    manifest=json.loads((folder/'source_manifest.json').read_text())
    for record in manifest['files']:
        assert digest(folder/'raw'/record['file'])==record['sha256'],record['file']
    stats=json.loads((folder/'processed'/'statistics.json').read_text())
    for f,h in stats['sha256'].items():assert digest(folder/'processed'/f)==h,f
    data=Dataset(folder)
    arrays=[data.train,data.validation,data.test]
    keys=[a[:,0]*data.n_items+a[:,1] for a in arrays]
    for key in keys:assert len(key)==len(np.unique(key))
    for i in range(3):
        for j in range(i):assert len(np.intersect1d(keys[i],keys[j]))==0
    for pairs in arrays[1:]:assert np.isin(pairs[:,1],data.supported_items).all()
    assert len(np.unique(data.train[:,0]))==data.n_users
    return {'dataset':name,'hashes_verified':True,'disjoint':True,'heldout_train_supported':True,
            'users':data.n_users,'items':data.n_items,'train':len(data.train),'validation':len(data.validation),'test':len(data.test)}

if __name__=='__main__':
    result=[check(name) for name in ['douban-book','yelp2018','amazon-book']]
    dest=ROOT/'docs'/'data_audit.json';dest.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
