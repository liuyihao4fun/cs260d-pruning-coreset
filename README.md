# Data-Efficient Recovery after Pruning via Coreset Selection

CS 260D (Large-Scale ML, Fall 2026) final project.

**Question:** after magnitude pruning, the network must be fine-tuned to recover accuracy.
How much of the training set does that fine-tuning really need, and which samples should we pick?

Pipeline: dense ResNet-18 (CIFAR-10) → global magnitude pruning → select a subset of the training
data (5–20%) → fine-tune on the subset only → test accuracy.

| method | idea |
|---|---|
| `full` | fine-tune on 100% of the data (upper bound) |
| `random` | class-balanced random subset (key baseline) |
| `el2n` | hardest samples for the pruned model, ‖softmax − onehot‖ |
| `craig` | CRAIG: facility location on last-layer gradients, weighted by cluster size |
| `damage_topk` | samples whose loss increased most due to pruning |
| `damage_fl` | **ours**: CRAIG objective where each sample's coverage is weighted by `alpha + damage_i / mean(damage)` |

## Quick start (Colab)

Open [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb) in Colab
(File → Open notebook → GitHub → `liuyihao4fun/cs260d-pruning-coreset`) and run the cells top to bottom.
Checkpoints and results are written to Google Drive (`MyDrive/cs260d/`).

The notebook embeds a copy of `src/`, so Colab needs no GitHub token. **After changing anything in
`src/`, run `python scripts/build_notebook.py` and commit the regenerated notebook too.**

## Command line

```bash
pip install -r requirements.txt

# 1. dense model (once)
python -m src.train_dense --epochs 40 --out checkpoints/resnet18_cifar10.pt

# 2. one experiment
python -m src.finetune --ckpt checkpoints/resnet18_cifar10.pt --sparsity 0.9 --method craig --frac 0.1

# 3. a grid (resumable; --shard/--num_shards split it across teammates)
python -m src.run_grid --ckpt checkpoints/resnet18_cifar10.pt --sparsities 0.9 \
    --methods full random craig el2n damage_fl --fracs 0.05 0.1 0.2 --seeds 0 1 2

# 4. table + figure
python -m src.plot --results results/results.jsonl --out results/
```

Debug on a laptop (CPU, tiny data): add `--limit_train 2000 --epochs 1`.

## Layout

```
src/
  data.py         CIFAR kept on the GPU + batched GPU augmentation (fast on Colab's 2 CPUs)
  models.py       ResNet-18 for 32x32 inputs
  prune.py        global magnitude pruning (torch.nn.utils.prune)
  train_dense.py  train the starting model
  finetune.py     prune -> select -> fine-tune -> evaluate; appends one JSON line per run
  run_grid.py     grid runner, skips finished runs, supports sharding
  plot.py         summary.csv + acc_vs_frac.png
  selection/      one file per method; add new ones to METHODS in __init__.py
notebooks/colab_run.ipynb   Colab launcher
```

## Team conventions
- Model weights and datasets never go into git (`.gitignore`); keep them on Drive.
- Each teammate writes to their own results file (`results_<name>.jsonl`); `plot.py` merges them.
- Pull before you push; keep commits small.
