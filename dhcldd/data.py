"""Deterministic preprocessing and train-only negative sampling."""
from pathlib import Path
import hashlib
import json
import numpy as np

PAPER_STATS = {'douban-book': [13024,22347,792062], 'yelp2018':[31668,38048,1561406], 'amazon-book':[52463,91599,2984108]}

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def load_source(folder):
    folder=Path(folder)
    if folder.name=='douban-book':
        raw=np.loadtxt(folder/'raw'/'user_book.dat',dtype=np.int64)
        if raw.ndim!=2 or raw.shape[1]!=3:raise ValueError('Expected user item rating triples')
        if not np.isin(raw[:,2],[1,2,3,4,5]).all():raise ValueError('Unexpected rating scale')
        stats={'rows':len(raw),'users':len(np.unique(raw[:,0])),'items':len(np.unique(raw[:,1])),
               'rating_histogram':{str(v):int((raw[:,2]==v).sum()) for v in range(1,6)},'threshold':4}
        return raw[raw[:,2]>=4,:2],stats
    chunks=[]
    for filename in ['train.txt','test.txt']:
        with (folder/'raw'/filename).open() as f:
            for line in f:
                ids=np.fromstring(line,sep=' ',dtype=np.int64)
                if len(ids)>1:chunks.append(np.column_stack((np.full(len(ids)-1,ids[0]),ids[1:])))
    pairs=np.concatenate(chunks)
    return pairs,{'rows':len(pairs),'users':len(np.unique(pairs[:,0])),'items':len(np.unique(pairs[:,1])),
                  'threshold':None,'note':'Union of upstream binarized train/test; original upstream partitions retained in raw/.'}

def k_core(pairs,k=5):
    pairs=np.unique(pairs,axis=0)
    while len(pairs):
        _,ui=np.unique(pairs[:,0],return_inverse=True)
        _,ii=np.unique(pairs[:,1],return_inverse=True)
        keep=(np.bincount(ui)[ui]>=k)&(np.bincount(ii)[ii]>=k)
        if keep.all():return pairs
        pairs=pairs[keep]
    raise ValueError('Empty k-core')

def preprocess(folder,seed=2026,k=5):
    folder=Path(folder)
    pairs,raw_stats=load_source(folder)
    qualified=len(pairs)
    pairs=k_core(pairs,k)
    users,ui=np.unique(pairs[:,0],return_inverse=True)
    items,ii=np.unique(pairs[:,1],return_inverse=True)
    pairs=np.column_stack((ui,ii)).astype(np.int64)
    rng=np.random.default_rng(seed)
    train=[];val=[];test=[]
    # Input pairs are unique and lexicographically ordered before randomization.
    cuts=np.r_[0,np.flatnonzero(np.diff(pairs[:,0]))+1,len(pairs)]
    for begin,end in zip(cuts[:-1],cuts[1:]):
        row=pairs[begin:end][rng.permutation(end-begin)]
        nv=max(1,int(np.floor(.1*len(row))))
        nt=max(1,int(np.floor(.2*len(row))))
        train.append(row[:len(row)-nv-nt]);val.append(row[len(row)-nv-nt:len(row)-nt]);test.append(row[len(row)-nt:])
    train=np.concatenate(train);val=np.concatenate(val);test=np.concatenate(test)
    supported=np.zeros(len(items),dtype=bool);supported[train[:,1]]=True
    excluded_val=int((~supported[val[:,1]]).sum());excluded_test=int((~supported[test[:,1]]).sum())
    val=val[supported[val[:,1]]];test=test[supported[test[:,1]]]
    out=folder/'processed';out.mkdir(parents=True,exist_ok=True)
    for name,array in [('train',train),('validation',val),('test',test),('interactions',pairs)]:
        np.savetxt(out/f'{name}.tsv',array,fmt='%d',delimiter='\t',header='user_id\titem_id',comments='')
    np.savetxt(out/'user_mapping.tsv',np.column_stack((np.arange(len(users)),users)),fmt='%d',delimiter='\t',header='user_id\tsource_user_id',comments='')
    np.savetxt(out/'item_mapping.tsv',np.column_stack((np.arange(len(items)),items)),fmt='%d',delimiter='\t',header='item_id\tsource_item_id',comments='')
    np.savez_compressed(out/'splits.npz',train=train,validation=val,test=test,n_users=len(users),n_items=len(items))
    stats={'dataset':folder.name,'split_seed':seed,'split_policy':'per-user floor(0.1*n) validation and floor(0.2*n) test, min 1; remainder train; exclude unsupported held-out items',
           'core':k,'upstream':raw_stats,'rating_qualified_rows':qualified,'users':len(users),'items':len(items),'interactions':len(pairs),
           'train':len(train),'validation':len(val),'test':len(test),'train_supported_items':int(supported.sum()),
           'excluded_validation_pairs':excluded_val,'excluded_test_pairs':excluded_test,
           'paper_reported_users_items_interactions':PAPER_STATS[folder.name],
           'matches_paper_statistics':[len(users),len(items),len(pairs)]==PAPER_STATS[folder.name],
           'provenance':'New deterministic split of a pinned public benchmark release; NOT the original paper split.',
           'sha256':{f.name:digest(f) for f in out.iterdir() if f.suffix in ['.tsv','.npz']}}
    (out/'statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
    return stats

class Dataset:
    def __init__(self,folder):
        import scipy.sparse as sp
        self.folder=Path(folder)
        with np.load(self.folder/'processed'/'splits.npz') as z:
            self.n_users=int(z['n_users']);self.n_items=int(z['n_items'])
            self.train=z['train'];self.validation=z['validation'];self.test=z['test']
        self.matrix=sp.csr_matrix((np.ones(len(self.train),dtype=np.float32),self.train.T),shape=(self.n_users,self.n_items))
        self.supported_items=np.flatnonzero(np.asarray(self.matrix.sum(axis=0)).ravel()>0)
        self.train_keys=np.sort(self.train[:,0]*self.n_items+self.train[:,1])
        if np.any(np.diff(self.matrix.indptr)>=self.n_items):raise ValueError('Cannot negative-sample saturated users')

    def negatives(self,users,rng):
        negatives=rng.integers(self.n_items,size=len(users))
        while True:
            keys=users*self.n_items+negatives
            loc=np.searchsorted(self.train_keys,keys)
            hit=(loc<len(self.train_keys))&(self.train_keys[np.minimum(loc,len(self.train_keys)-1)]==keys)
            if not hit.any():return negatives
            negatives[hit]=rng.integers(self.n_items,size=int(hit.sum()))
