"""Re-evaluate a trusted saved model on its exact recorded split."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from dhcldd.data import Dataset,digest
from dhcldd.graph import build_graph
from dhcldd.model import DHCLDD
from dhcldd.evaluate import evaluate

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--device',default='auto')
    p.add_argument('--split',choices=['validation','test'],default='test');p.add_argument('--output',required=True)
    args=p.parse_args()
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    state=torch.load(args.checkpoint,map_location=device,weights_only=False);c=state['config']
    torch.set_num_threads(c.get('cpu_threads',4))
    data=Dataset(ROOT/'data'/c['dataset'])
    if digest(data.folder/'processed'/'splits.npz')!=state['split_sha256']:raise ValueError('Checkpoint and data split do not match')
    key=hashlib.sha256(data.train.tobytes()+str(c['q']).encode()).hexdigest()[:20]
    graph=build_graph(data.train,data.n_users,data.n_items,c['q'],device,data.folder/'cache'/f'original_{key}.npz')
    model=DHCLDD(data.n_users,data.n_items,c['d'],c['layers']).to(device)
    model.load_state_dict(state['model'])
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    metrics=evaluate(model,graph,data,args.split,c['eval_batch_size'],out.with_suffix('.csv'))
    out.write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
