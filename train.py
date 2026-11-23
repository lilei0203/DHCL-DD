"""Train a reconstructed DHCL-DD model, select on validation, then test once."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from dhcldd.data import Dataset,digest
from dhcldd.graph import build_graph,filter_interactions
from dhcldd.model import DHCLDD,contrastive_loss
from dhcldd.evaluate import evaluate

ROOT=Path(__file__).resolve().parent

def read_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/paper.json')
    p.add_argument('--dataset',choices=['douban-book','yelp2018','amazon-book'],default='douban-book')
    p.add_argument('--device',default='auto');p.add_argument('--run-dir',default=None)
    p.add_argument('--epochs',type=int);p.add_argument('--batch-size',type=int)
    p.add_argument('--seed',type=int);p.add_argument('--resume',default=None)
    p.add_argument('--profile-steps',type=int,default=0,help='Profile warm-up and joint updates; does not produce a final evaluation')
    args=p.parse_args()
    c=json.loads((ROOT/args.config).read_text())
    for key in ['epochs','batch_size','seed']:
        value=getattr(args,key)
        if value is not None:c[key]=value
    c.update(dataset=args.dataset,device=args.device,config=args.config)
    if c['epochs']<=c['warmup']:raise ValueError('epochs must extend past warm-up')
    return args,c

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False

def train_step(model,original,filtered,users,positives,negatives,c,optimizer):
    model.train();optimizer.zero_grad(set_to_none=True)
    n=model.n_users
    f=model.encode(original)
    margins=(f[users]*(f[positives+n]-f[negatives+n])).sum(-1)
    bpr=F.softplus(-margins).mean()
    reg=model.regularization(torch.cat([users,positives+n,negatives+n]))
    cl=f.new_zeros(())
    if filtered is not None:
        ff=model.encode(filtered)
        cl=contrastive_loss(f,ff,users,torch.cat([positives+n,negatives+n]),c['tau'],c['contrastive_chunk'])
    loss=bpr+c['lambda_reg']*reg+c['lambda_cl']*cl
    if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
    loss.backward();optimizer.step()
    return {name:float(value.detach()) for name,value in [('loss',loss),('bpr',bpr),('cl',cl),('regularization',reg)]}

def main():
    args,c=read_args();seed_all(c['seed']);torch.set_num_threads(c.get('cpu_threads',4))
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    c['device']=device
    data=Dataset(ROOT/'data'/args.dataset)
    run=Path(args.run_dir) if args.run_dir else ROOT/'runs'/f"{args.dataset}_{Path(args.config).stem}_seed{c['seed']}"
    if not run.is_absolute():run=ROOT/run
    run.mkdir(parents=True,exist_ok=True)
    if (run/'history.jsonl').exists() and not args.resume and not args.profile_steps:
        raise FileExistsError('Existing run: use a new --run-dir or --resume; logs are never silently overwritten')
    stats=json.loads((data.folder/'processed'/'statistics.json').read_text())
    env={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,
         'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name() if device=='cuda' else None,
         'device':device,'split_sha256':digest(data.folder/'processed'/'splits.npz'),
         'code_sha256':{str(f.relative_to(ROOT)):digest(f) for f in [ROOT/'train.py',*sorted((ROOT/'dhcldd').glob('*.py'))]},
         'status':'New independent reconstruction run; not original manuscript experimental evidence',
         'determinism':'Seeded preprocessing, sampling, initialization; CUDA index_add/scatter floating-point reductions can be nondeterministic.'}
    (run/'config.json').write_text(json.dumps(c,indent=2));(run/'environment.json').write_text(json.dumps(env,indent=2))
    key=hashlib.sha256(data.train.tobytes()+str(c['q']).encode()).hexdigest()[:20]
    print('Building exact train-only overlap graph...',flush=True)
    original=build_graph(data.train,data.n_users,data.n_items,c['q'],device,data.folder/'cache'/f'original_{key}.npz')
    print(json.dumps(original.statistics),flush=True)
    (run/'original_graph.json').write_text(json.dumps(original.statistics,indent=2))
    model=DHCLDD(data.n_users,data.n_items,c['d'],c['layers'],c['checkpoint_layers']).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=c['lr'])
    rng=np.random.default_rng(c['seed']);start_epoch=1;best=-float('inf');bad=0;best_epoch=None
    if args.resume:
        if Path(args.resume).resolve().parent != run.resolve():
            raise ValueError('--run-dir must be the existing checkpoint directory when resuming')
        state=torch.load(args.resume,map_location=device,weights_only=False)
        if state['split_sha256']!=env['split_sha256']:raise ValueError('Resume split mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state=state['numpy_generator_state'];torch.set_rng_state(state['torch_rng'].cpu())
        if device=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        start_epoch=state['epoch']+1;best=state['best'];bad=state['bad'];best_epoch=state['best_epoch']
        if start_epoch>c['epochs']:raise ValueError('This run already reached the requested epoch limit')
    if args.profile_steps:
        profile={}
        for phase in ['warmup','joint']:
            filtered=None
            if phase=='joint':
                model.eval()
                with torch.no_grad():snap=model.encode(original)
                retained,_=filter_interactions(data.train,snap,data.n_users,c['gamma'],c['k_min']);del snap
                filtered=build_graph(retained,data.n_users,data.n_items,c['q'],device)
            t=time.perf_counter();losses=[]
            if device=='cuda':torch.cuda.reset_peak_memory_stats()
            for _ in range(args.profile_steps):
                pairs=data.train[rng.choice(len(data.train),min(c['batch_size'],len(data.train)),replace=False)]
                neg=data.negatives(pairs[:,0],rng)
                u,p,n=(torch.as_tensor(a,device=device) for a in [pairs[:,0],pairs[:,1],neg])
                losses.append(train_step(model,original,filtered,u,p,n,c,optimizer))
            profile[phase]={'seconds_per_step':(time.perf_counter()-t)/args.profile_steps,'losses':losses,
                            'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0}
            print(json.dumps({phase:profile[phase]}),flush=True)
        (run/'profile.json').write_text(json.dumps(profile,indent=2));return
    total_start=time.perf_counter()
    for epoch in range(start_epoch,c['epochs']+1):
        t=time.perf_counter();filtered=None;filter_stats=None
        if epoch>c['warmup']:
            model.eval()
            with torch.no_grad():snapshot=model.encode(original)
            retained,filter_stats=filter_interactions(data.train,snapshot,data.n_users,c['gamma'],c['k_min']);del snapshot
            filtered=build_graph(retained,data.n_users,data.n_items,c['q'],device)
            np.savez_compressed(run/'last_filtered_interactions.npz',pairs=retained,epoch=epoch)
            print(f'Epoch {epoch} filtered graph: {filter_stats}',flush=True)
        permutation=rng.permutation(len(data.train));sums={k:0. for k in ['loss','bpr','cl','regularization']};seen=0;steps=0
        for begin in range(0,len(permutation),c['batch_size']):
            pairs=data.train[permutation[begin:begin+c['batch_size']]]
            neg=data.negatives(pairs[:,0],rng)
            u,p,n=(torch.as_tensor(a,device=device) for a in [pairs[:,0],pairs[:,1],neg])
            losses=train_step(model,original,filtered,u,p,n,c,optimizer)
            for k in sums:sums[k]+=losses[k]*len(pairs)
            seen+=len(pairs);steps+=1
            if steps%25==0:print(f'Epoch {epoch} step {steps}: BPR {losses["bpr"]:.6f}, CL {losses["cl"]:.4f}',flush=True)
        record={'epoch':epoch,'phase':'warmup' if filtered is None else 'joint','examples':seen,'steps':steps,
                **{k:v/seen for k,v in sums.items()},'filter':filter_stats,'training_seconds':time.perf_counter()-t}
        check=epoch%c['eval_every']==0 or epoch==c['epochs']
        if check:
            metrics=evaluate(model,original,data,'validation',c['eval_batch_size'])
            record['validation']=metrics
            if metrics['NDCG@20']>best:
                best=metrics['NDCG@20'];best_epoch=epoch;bad=0
                torch.save({'model':model.state_dict(),'epoch':epoch,'config':c,'validation':metrics,
                            'split_sha256':env['split_sha256']},run/'best.pt')
            elif epoch>c['warmup']:bad+=1
        with (run/'history.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'config':c,
               'best':best,'best_epoch':best_epoch,'bad':bad,'numpy_generator_state':rng.bit_generator.state,
               'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else [],
               'split_sha256':env['split_sha256']}
        torch.save(state,run/'last.pt')
        print(json.dumps(record),flush=True)
        if check and epoch>c['warmup'] and bad>=c['patience']:break
    selected=torch.load(run/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(selected['model'])
    result=evaluate(model,original,data,'test',c['eval_batch_size'],run/'test_per_user.csv')
    result.update(best_epoch=best_epoch,best_validation_ndcg20=best,completed_epochs=epoch,
                  elapsed_seconds=time.perf_counter()-total_start,train_interactions=len(data.train),
                  run_scope=c.get('run_scope','paper schedule'),paper_results_reproduced=False,
                  note='Measured on a newly constructed split; compare only with baselines rerun on this identical split.')
    (run/'results.json').write_text(json.dumps(result,indent=2))
    print('FINAL '+json.dumps(result),flush=True)

if __name__=='__main__':main()
