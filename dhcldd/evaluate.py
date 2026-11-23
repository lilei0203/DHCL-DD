"""Full-catalog evaluation with manuscript candidate masking and per-user output."""
import csv
import numpy as np
import torch

def per_user_metrics(recommendations,truth,ks=(10,20)):
    hits=np.isin(recommendations,truth).astype(np.float64)
    result={}
    for k in ks:
        size=min(k,len(recommendations))
        discount=1/np.log2(np.arange(size)+2)
        ideal=(1/np.log2(np.arange(min(k,len(truth)))+2)).sum()
        result[f'Recall@{k}']=float(hits[:size].sum()/len(truth))
        result[f'NDCG@{k}']=float((hits[:size]*discount).sum()/ideal)
    return result

@torch.no_grad()
def evaluate(model,graph,data,split='validation',batch_size=128,output=None):
    if split not in ['validation','test']:raise ValueError(split)
    model.eval();emb=model.encode(graph)
    n=data.n_users
    truth_pairs=getattr(data,split)
    histories=[set() for _ in range(n)];truth=[[] for _ in range(n)]
    for u,i in data.train:histories[u].add(int(i))
    if split=='test':
        for u,i in data.validation:histories[u].add(int(i))
    for u,i in truth_pairs:truth[u].append(int(i))
    users=[u for u in range(n) if truth[u]]
    supported=torch.zeros(data.n_items,dtype=torch.bool,device=emb.device)
    supported[torch.as_tensor(data.supported_items,device=emb.device)]=True
    records=[]
    for start in range(0,len(users),batch_size):
        us=users[start:start+batch_size]
        score=emb[us]@emb[n:].T
        score[:,~supported]=-torch.inf
        mask_rows=[];mask_items=[]
        for row,u in enumerate(us):
            mask_rows.extend([row]*len(histories[u]));mask_items.extend(histories[u])
        score[mask_rows,mask_items]=-torch.inf
        # Stable item-index tie-break; complete catalog, no sampled negatives.
        order=torch.argsort(score,dim=1,descending=True,stable=True)[:,:20]
        ranked=order.cpu().numpy()
        finite=torch.gather(score,1,order).isfinite().cpu().numpy()
        for row,u in enumerate(us):
            rec=ranked[row][finite[row]]
            record={'user_id':u,'positives':len(truth[u]),**per_user_metrics(rec,truth[u])}
            records.append(record)
    metrics={k:float(np.mean([r[k] for r in records])) if records else 0.0 for k in ['Recall@10','NDCG@10','Recall@20','NDCG@20']}
    metrics.update(evaluated_users=len(records),excluded_users=n-len(records),split=split)
    if output:
        with open(output,'w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=['user_id','positives','Recall@10','NDCG@10','Recall@20','NDCG@20'])
            writer.writeheader();writer.writerows(records)
    return metrics
