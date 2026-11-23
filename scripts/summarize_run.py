"""Create an evidence summary only from completed on-disk measurements."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    run=ROOT/'runs'/'douban-book_local_run_seed2026'
    result=json.loads((run/'results.json').read_text())
    config=json.loads((run/'config.json').read_text())
    history=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
    stats={name:json.loads((ROOT/'data'/name/'processed'/'statistics.json').read_text()) for name in ['douban-book','yelp2018','amazon-book']}
    assert len(history)==result['completed_epochs']
    assert all(row['examples']==stats['douban-book']['train'] for row in history)
    assert any(row['phase']=='joint' and row['cl']>0 for row in history)
    report={'experiment_status':'Completed short full-data reconstruction experiment; not a reproduction of the manuscript performance table',
            'datasets':{k:{field:v[field] for field in ['users','items','interactions','train','validation','test','matches_paper_statistics']} for k,v in stats.items()},
            'trained_dataset':'douban-book','config':config,'measured_test_results':result,
            'full_training_set_traversed_each_epoch':True,'joint_training_epochs':[r['epoch'] for r in history if r['phase']=='joint'],
            'selected_checkpoint_phase':next(r['phase'] for r in history if r['epoch']==result['best_epoch']),
            'original_experiment_code_available':False,'original_data_split_available':False,
            'public_repository_created':False,'doi_minted':False}
    (ROOT/'docs'/'RUN_REPORT.json').write_text(json.dumps(report,indent=2))
    text=f'''# 本次实际运行结果

已完成 Douban-Book 的完整处理后数据短程实验，未训练另外两个数据集。结果来自实际日志和保存模型，未使用论文表格或估读图片数值。

| 项目 | 实际值 |
|---|---:|
| 训练交互数 | {result['train_interactions']:,} |
| 完成轮数 | {result['completed_epochs']} |
| 最佳验证轮次 | {result['best_epoch']} |
| 测试用户数 | {result['evaluated_users']:,} |
| Recall@10 | {result['Recall@10']:.8f} |
| NDCG@10 | {result['NDCG@10']:.8f} |
| Recall@20 | {result['Recall@20']:.8f} |
| NDCG@20 | {result['NDCG@20']:.8f} |
| 运行时间（秒） | {result['elapsed_seconds']:.1f} |

训练设置为 64 维、2 层、10 轮预热后启用动态过滤及对比学习。每轮都遍历全部训练正例。按验证 NDCG@20 选择模型后进行全物品排序测试；逐用户结果在 `runs/douban-book_local_run_seed2026/test_per_user.csv`。

本次最佳验证模型来自第 10 轮预热阶段。第 12 轮联合训练的验证 NDCG@20 为 0.03707622，低于第 10 轮的 0.03827876，因此按规则选择第 10 轮模型。上述测试指标对应这一预热模型，不应作为对比学习有效性的证据；虽已运行两轮联合训练，尚不能声称其优于预热模型。

这是新实现、新划分上的单种子短程实验，batch size 为 {config['batch_size']}，尚未完成论文的五种子、最多 500 轮及基线对比，不能将本次指标替换进原比较表并沿用其结论。论文数据统计的差异见 `docs/IMPLEMENTATION_NOTES.md`。本地项目尚未公开发布，也没有生成 DOI。
'''
    (ROOT/'运行结果.md').write_text(text,encoding='utf8')
    print(json.dumps(report,indent=2))
