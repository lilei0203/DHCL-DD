"""Check the delivered run's internal evidence and numerical integrity."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dhcldd.data import digest,Dataset

if __name__=='__main__':
    run=ROOT/'runs'/'douban-book_local_run_seed2026'
    env=json.loads((run/'environment.json').read_text())
    result=json.loads((run/'results.json').read_text())
    history=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
    for name,h in env['code_sha256'].items():assert digest(ROOT/name.replace('\\','/'))==h,name
    for row in history:
        assert row['examples']==result['train_interactions']
        assert all(np.isfinite(row[k]) for k in ['loss','bpr','cl','regularization'])
    assert [row['epoch'] for row in history]==list(range(1,result['completed_epochs']+1))
    validations=[r for r in history if 'validation' in r]
    selected=max(validations,key=lambda r:r['validation']['NDCG@20'])
    assert selected['epoch']==result['best_epoch']
    checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    assert checkpoint['epoch']==result['best_epoch']
    assert checkpoint['split_sha256']==env['split_sha256']
    assert all(torch.isfinite(v).all().item() for v in checkpoint['model'].values())
    data=Dataset(ROOT/'data'/'douban-book')
    with np.load(run/'last_filtered_interactions.npz') as filtered_file:
        retained=filtered_file['pairs'];filtered_epoch=int(filtered_file['epoch'])
    original_keys=data.train[:,0]*data.n_items+data.train[:,1]
    retained_keys=retained[:,0]*data.n_items+retained[:,1]
    assert np.isin(retained_keys,original_keys).all()
    assert np.array_equal(np.unique(retained[:,0]),np.unique(data.train[:,0]))
    assert np.array_equal(np.unique(retained[:,1]),data.supported_items)
    with (run/'test_per_user.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==result['evaluated_users']
    for metric in ['Recall@10','NDCG@10','Recall@20','NDCG@20']:
        mean=np.mean([float(row[metric]) for row in rows])
        assert abs(mean-result[metric])<1e-12,metric
    report={'code_hashes_match_executed_version':True,'all_weights_and_losses_finite':True,
            'every_epoch_covered_full_training_set':True,'checkpoint_selected_by_validation_only':True,
            'per_user_test_means_match_summary':True,'completed_epochs':len(history),
            'last_filtered_epoch':filtered_epoch,'filtered_edges_subset_of_training':True,
            'filtered_training_user_and_item_coverage_preserved':True,
            'joint_epochs':sum(r['phase']=='joint' for r in history),'test_users':len(rows)}
    (ROOT/'docs'/'run_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
