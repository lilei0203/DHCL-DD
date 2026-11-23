import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dhcldd.data import preprocess

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='all',choices=['all','douban-book','yelp2018','amazon-book'])
    p.add_argument('--seed',type=int,default=2026);args=p.parse_args()
    for name in (['douban-book','yelp2018','amazon-book'] if args.dataset=='all' else [args.dataset]):
        result=preprocess(ROOT/'data'/name,args.seed)
        print(json.dumps(result,indent=2),flush=True)
