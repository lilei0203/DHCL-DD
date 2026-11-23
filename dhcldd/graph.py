"""Train-only hypergraph structures and exact top-Jaccard neighborhoods."""
from dataclasses import dataclass
import time
import numpy as np
import scipy.sparse as sp
import torch

@dataclass
class Hypergraph:
    vertices: torch.Tensor
    edges: torch.Tensor
    query: torch.Tensor
    neighbor: torch.Tensor
    n_vertices: int
    n_edges: int
    statistics: dict

def exact_neighbors(incidence,q,block_size=256):
    """Blocked sparse H.T@H: enumerate ALL overlapping candidates, keep q+self.

    Blocking bounds temporary memory; this is not approximate neighbor sampling.
    """
    h=incidence.astype(np.float32).tocsr()
    ht=h.T.tocsr();degree=np.asarray(h.sum(axis=0)).ravel()
    queries=[];neighbors=[];candidate_count=0
    for start in range(0,h.shape[1],block_size):
        overlap=(ht[start:start+block_size]@h).tocsr()
        for local in range(overlap.shape[0]):
            e=start+local
            a,b=overlap.indptr[local:local+2]
            ids=overlap.indices[a:b];count=overlap.data[a:b]
            mask=ids!=e;ids=ids[mask];count=count[mask];candidate_count+=len(ids)
            if len(ids):
                jaccard=count/(degree[e]+degree[ids]-count)
                # Exact ties are resolved by ascending fixed edge index.
                order=np.lexsort((ids,-jaccard))[:q]
                chosen=np.r_[e,ids[order]]
            else:chosen=np.array([e])
            queries.append(np.full(len(chosen),e,dtype=np.int64));neighbors.append(chosen)
    return np.concatenate(queries),np.concatenate(neighbors),candidate_count

def build_graph(pairs,n_users,n_items,q=16,device='cpu',cache=None):
    start=time.perf_counter()
    vertices=np.r_[pairs[:,0],pairs[:,1]+n_users]
    edges=np.r_[pairs[:,1],pairs[:,0]+n_items]
    n=n_users+n_items
    h=sp.csr_matrix((np.ones(len(vertices),dtype=np.float32),(vertices,edges)),shape=(n,n))
    if cache is not None and cache.exists():
        with np.load(cache) as z:query=z['query'];neighbor=z['neighbor'];candidate_count=int(z['candidate_count'])
    else:
        query,neighbor,candidate_count=exact_neighbors(h,q)
        if cache is not None:
            cache.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(cache,query=query,neighbor=neighbor,candidate_count=candidate_count)
    return Hypergraph(*(torch.tensor(x,dtype=torch.long,device=device) for x in [vertices,edges,query,neighbor]),
                      n,n,{'interactions':len(pairs),'incidences':len(vertices),'neighbor_pairs':len(query),
                           'overlapping_candidates':candidate_count,'construction_seconds':time.perf_counter()-start})

def filter_interactions(pairs,embeddings,n_users,gamma=.8,k_min=2):
    """Detached cosine snapshot, per-user top-k, then item coverage restoration."""
    with torch.no_grad():
        scores=[]
        for begin in range(0,len(pairs),65536):
            p=pairs[begin:begin+65536]
            u=embeddings[torch.as_tensor(p[:,0],device=embeddings.device)]
            i=embeddings[torch.as_tensor(p[:,1]+n_users,device=embeddings.device)]
            s=(u*i).sum(-1)/(u.norm(dim=-1)*i.norm(dim=-1)).clamp_min(1e-8)
            scores.append((.5*(1+s)).clamp(0,1).cpu().numpy())
        scores=np.concatenate(scores)
    degree=np.bincount(pairs[:,0],minlength=n_users)
    budget=np.minimum(degree,np.maximum(k_min,np.ceil(gamma*degree).astype(np.int64)))
    order=np.lexsort((pairs[:,1],-scores,pairs[:,0]))
    sorted_users=pairs[order,0]
    start=np.maximum.accumulate(np.where(np.r_[True,np.diff(sorted_users)!=0],np.arange(len(order)),0))
    take=np.arange(len(order))-start<budget[sorted_users]
    keep=np.zeros(len(pairs),dtype=bool);keep[order[take]]=True
    supported=np.unique(pairs[:,1]);covered=np.unique(pairs[keep,1]);missing=np.setdiff1d(supported,covered)
    if len(missing):
        # The first occurrence per item has highest score, then smallest user ID.
        by_item=np.lexsort((pairs[:,0],-scores,pairs[:,1]))
        first=np.r_[True,np.diff(pairs[by_item,1])!=0]
        winners=by_item[first];keep[winners[np.isin(pairs[winners,1],missing)]]=True
    return pairs[keep],{'retained':int(keep.sum()),'original':len(pairs),'actual_retention':float(keep.mean()),
                        'restored_items':len(missing),'min_retained_user_degree':int(np.bincount(pairs[keep,0],minlength=n_users)[degree>0].min())}
