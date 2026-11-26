"""Run a grid of (sparsity x method x frac x seed) experiments, resumably.

    python -m src.run_grid --ckpt /content/drive/MyDrive/cs260d/resnet18_cifar10.pt \
        --results /content/drive/MyDrive/cs260d/results.jsonl \
        --sparsities 0.9 --methods random craig el2n damage_fl --fracs 0.05 0.1 0.2 --seeds 0 1 2

* Only completed, verified v2 identities in --results are skipped. After a Colab
  disconnect, re-run the same command/commit; incomplete runs restart from dense.
* --shard i --num_shards 3 splits the grid into three independent shards:
  each person runs a different shard into their own results file. Never use
  concurrent writers in different Colab runtimes for the same Drive journal.
* 'full' (100% data) is run once per sparsity/seed regardless of --fracs.
* The dataset is loaded onto the GPU once and reused by every run.
"""
import argparse
import itertools

from .data import load_cifar
from .finetune import ALL_METHODS, build_parser, run
from .utils import get_device, load_jsonl
from .audit import completed_keys, experiment_identity, file_sha256, identity_key, source_sha256


def main():
    # allow_abbrev=False so finetune flags like --seed are not mistaken for --seeds
    g = argparse.ArgumentParser(allow_abbrev=False)
    g.add_argument("--sparsities", type=float, nargs="+", default=[0.9])
    g.add_argument("--methods", nargs="+", default=["full", "random", "craig", "el2n", "damage_fl"],
                   choices=ALL_METHODS)
    g.add_argument("--fracs", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    g.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    g.add_argument("--shard", type=int, default=0)
    g.add_argument("--num_shards", type=int, default=1)
    g.add_argument("--dry_run", action="store_true", help="only list the runs")
    grid_args, rest = g.parse_known_args()
    base = build_parser().parse_args(rest + ["--method", "random"])   # shared settings

    configs = []
    for sp, m, seed in itertools.product(grid_args.sparsities, grid_args.methods, grid_args.seeds):
        for fr in ([1.0] if m == "full" else grid_args.fracs):
            a = argparse.Namespace(**vars(base))
            a.sparsity, a.method, a.frac, a.seed = sp, m, fr, seed
            configs.append(a)
    if grid_args.num_shards < 1 or not 0 <= grid_args.shard < grid_args.num_shards:
        g.error("require num_shards >= 1 and 0 <= shard < num_shards")
    configs = configs[grid_args.shard::grid_args.num_shards]

    records = load_jsonl(base.results)
    done = completed_keys(records)
    checkpoint_hash, source_hash = file_sha256(base.ckpt), source_sha256()
    keyed = [(a, identity_key(experiment_identity(a, checkpoint_hash, source_hash))) for a in configs]
    todo = [(a, key) for a, key in keyed if key not in done]
    legacy = sum(r.get("schema_version") != 2 and "final_acc" in r for r in records)
    if legacy:
        print(f"{legacy} legacy completions retained; missing v2 provenance, so they do not skip new runs")
    print(f"{len(configs)} runs in this shard, {len(configs) - len(todo)} already done, {len(todo)} to go")
    if grid_args.dry_run:
        for a, key in todo:
            print(f"  {key} sp={a.sparsity} method={a.method} frac={a.frac} epochs={a.epochs} seed={a.seed} rs={a.reselect_every} uniform={a.no_weights}")
        return

    if not todo:
        return

    data = load_cifar(base.dataset, base.data_root, get_device(), base.limit_train)
    for i, (a, _) in enumerate(todo, 1):
        print(f"\n=== run {i}/{len(todo)} ===")
        run(a, data=data)


if __name__ == "__main__":
    main()
