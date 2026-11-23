# DHCL-DD

Independent PyTorch reconstruction of **Dynamic hypergraph contrastive learning with dual-level denoising for recommendation**, based on the supplied manuscript. This project contains a working implementation, pinned public benchmark inputs, deterministic newly generated splits, tests, and recorded local experiments.

## Layout

```text
DHCL-DD/
  dhcldd/                 model, structures, preprocessing, full-ranking evaluator
  configs/paper.json      manuscript default hyperparameters and 500-epoch limit
  configs/local_run.json  explicitly shortened local experiment configuration
  scripts/               download, preprocessing and integrity checks
  tests/                 numerical reference, gradient and leakage tests
  data/
    douban-book/
    yelp2018/
    amazon-book/
      raw/               unchanged upstream files
      processed/         interactions, train/validation/test TSVs, ID maps, NPZ
      source_manifest.json
  runs/                  logs, configurations, checkpoints and measured metrics
  docs/                  implementation choices and manuscript discrepancies
```

## Environment

Python 3.12 was used. Install in a dedicated virtual environment. For the local NVIDIA driver (CUDA 12.1 compatible), install PyTorch from its official CUDA 12.1 index:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

CPU-only installation is supported with `--index-url https://download.pytorch.org/whl/cpu` in the first pip command. Linux uses `.venv/bin/python` instead. See [official PyTorch installation history](https://pytorch.org/get-started/previous-versions/). A 4GB GPU is adequate only with careful memory use; large datasets and the full paper schedule require substantially more runtime. Layer activation checkpointing and chunked exact contrastive denominators reduce memory without caching stale trainable representations.

If Windows reports WinError 650 when installing through a junction, invoke the Python executable by its resolved physical path. The current workspace's `C:\Users\lilei\Documents\Codex\...` location resolves to `D:\codex\outputs\...`.

## Download and prepare all three datasets

```powershell
.venv\Scripts\python.exe scripts/download_data.py
.venv\Scripts\python.exe scripts/prepare_data.py --seed 2026
.venv\Scripts\python.exe scripts/check_data.py
```

The files are already downloaded and prepared in this delivered local project. The scripts make the process repeatable on another machine. Raw source files are never replaced by invented samples. Sources are pinned to commits with SHA-256 checksums:

| Dataset | Upstream source | Treatment |
|---|---|---|
| Douban-Book | [HIN datasets maintained by librahu](https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding/tree/cc05c484b628b08237f0a876fcebf99d736ccbea/Douban%20Book) | Explicit ratings >=4, deduplication, iterative 5-core |
| Yelp2018 | [LightGCN authors' release](https://github.com/kuandeng/LightGCN/tree/b067ec05f16a24e0e1efe63cd82385f0c469af20/Data/yelp2018) | Already binarized; union upstream train/test, deduplicate, 5-core, new split |
| Amazon-Book | [LightGCN authors' release](https://github.com/kuandeng/LightGCN/tree/b067ec05f16a24e0e1efe63cd82385f0c469af20/Data/amazon-book) | Already binarized; union upstream train/test, deduplicate, 5-core, new split |

**Dataset statistics in the supplied manuscript are inconsistent with its stated preprocessing.** In particular, Douban's 792,062 interactions are the unfiltered rating count. The actual positive 5-core and new split are recorded in `statistics.json`. 

## Verify the implementation

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests compare sparse attention and gradients against direct equation calculations, exact Jaccard neighbors and tie-breaking, structural coverage, exact symmetric InfoNCE, ranking metrics, and disjoint deterministic splits.

## Run one dataset

```powershell
# Manuscript defaults, one seed; validation every 5 epochs, max 500 epochs
.venv\Scripts\python.exe -u train.py --dataset douban-book --config configs/paper.json

# Short full-data local run: 10 warm-up + 2 joint epochs, larger BPR batches
.venv\Scripts\python.exe -u train.py --dataset douban-book --config configs/local_run.json
```

Use `--run-dir runs/new_name` for an additional run. Existing histories are protected from accidental overwrite. Resume interrupted training with:

- `config.json`, `environment.json`: actual settings, software, hardware, split and code hashes.
- `history.jsonl`: per-epoch actual loss, complete interaction count, filtering statistics, validation results and elapsed time.
- `best.pt`, `last.pt`: best validation model and resumable last training state.
- `results.json`, `test_per_user.csv`: measured full-ranking test results and per-user values.


