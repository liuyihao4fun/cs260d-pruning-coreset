"""Prune a dense checkpoint, select a training subset, fine-tune, evaluate.

Single run:
    python -m src.finetune --ckpt checkpoints/resnet18_cifar10.pt \
        --sparsity 0.9 --method craig --frac 0.1 --seed 0

Each run appends one JSON line to --results (default results/results.jsonl).
Many runs: see src/run_grid.py.
"""
import argparse
import platform
from pathlib import Path
import time
import uuid
import warnings

import torch
import torch.nn.functional as F

from .data import GpuLoader, NUM_CLASSES, load_cifar
from .models import resnet18
from .prune import global_magnitude_prune, measure_sparsity
from .selection import METHODS, NEEDS_DENSE, SelectionContext, select
from .audit import (SCHEMA_VERSION, completed_keys, experiment_identity, full_config,
                    git_metadata, identity_key, load_jsonl, run_key, utc_now)
from .utils import (append_jsonl, autocast, evaluate, get_device, per_sample_outputs,
                    set_seed, synchronize, timed)

ALL_METHODS = ["full"] + list(METHODS)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="dense checkpoint from src.train_dense")
    p.add_argument("--dataset", default="cifar10", choices=list(NUM_CLASSES))
    p.add_argument("--sparsity", type=float, default=0.9)
    p.add_argument("--method", default="random", choices=ALL_METHODS)
    p.add_argument("--frac", type=float, default=0.1, help="fraction of training data (ignored for 'full')")
    p.add_argument("--epochs", type=int, default=30, help="fine-tuning epochs over the subset")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reselect_every", type=int, default=0,
                   help="re-run selection every R epochs with the current model (0 = select once)")
    p.add_argument("--no_weights", action="store_true", help="ignore CRAIG-style per-sample weights")
    p.add_argument("--damage_alpha", type=float, default=1.0, help="alpha for damage_fl")
    p.add_argument("--results", default="results/results.jsonl")
    p.add_argument("--tag", default="", help="free-form label stored with the result")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--limit_train", type=int, default=None, help="debug: use first N images")
    return p


def load_dense(a, device):
    ck = torch.load(a.ckpt, map_location=device)
    model = resnet18(NUM_CLASSES[a.dataset]).to(device)
    model.load_state_dict(ck["state_dict"])
    return model


def weight_stats(w, probs, y, epoch):
    """Diagnostics for one selection: how skewed the per-sample training weights are,
    and how well the current model already fits the full training set."""
    w = w.float()
    top = w.sort(descending=True).values
    k1 = max(1, len(w) // 100)
    s = {"epoch": epoch, "w_max_over_mean": round(float(w.max() / w.mean()), 1),
         "w_top1pct_share": round(float(top[:k1].sum() / w.sum()), 3)}
    if probs is not None:
        s["train_acc_before_select"] = round(float((probs.argmax(1) == y).float().mean()) * 100, 2)
    return s


def make_subset(a, model, data, dense_loss, device, timings, epoch=0):
    """Return indices, weights and diagnostics; time scoring separately.

    `epoch` offsets the random generator so that re-selection (random method) draws a
    new subset each time; epoch 0 keeps the original seed, so one-shot runs are unchanged.
    """
    x_tr, y_tr = data[0], data[1]
    if a.method == "full":
        idx = torch.arange(len(y_tr), device=device)
        return idx, torch.ones(len(idx), device=device), {}
    loss = probs = feats = None
    if a.method != "random":
        with timed(timings, "pruned_scoring_time", device):
            loss, probs, feats = per_sample_outputs(model, x_tr, y_tr, a.dataset, device)
    ctx = SelectionContext(y=y_tr, num_classes=NUM_CLASSES[a.dataset], loss=loss, probs=probs,
                           feats=feats, dense_loss=dense_loss, damage_alpha=a.damage_alpha,
                           generator=torch.Generator().manual_seed(a.seed + 1000 * epoch))
    with timed(timings, "selection_time", device):
        idx, w = select(a.method, ctx, a.frac)
        if a.no_weights:
            w = torch.ones_like(w)
    return idx, w, weight_stats(w, probs, y_tr, epoch)


def run(a, data=None, verbose=True):
    """Journal one attempt. Incomplete attempts restart from the dense checkpoint.

    Dense losses are computed once per attempt, kept in memory for re-selection,
    and never reused across runs. Every attempt pays its own dense scoring cost.
    """
    start = time.perf_counter()
    started_at = utc_now()
    identity = experiment_identity(a)
    key = identity_key(identity)
    if key in completed_keys(load_jsonl(a.results)):
        if verbose:
            print(f"[{key}] already completed; skipping", flush=True)
        return None
    device = get_device()
    metadata = {
        "git": git_metadata(), "checkpoint_path": str(Path(a.ckpt).resolve()),
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "source_sha256": identity["source_sha256"], "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_model": torch.cuda.get_device_name() if device == "cuda" else None,
        "device": device, "seed": a.seed,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "dense_scoring_cache": "in-memory within this attempt only; no cross-run cache",
        "data_preloaded": data is not None,
        "wall_clock_scope": ("run entry through final evaluation/metadata; excludes final journal append; "
                             + ("shared data loading excluded" if data is not None else "standalone data loading included")),
        "timing_schema": "v2-disjoint-scoring-selection-training-evaluation",
        "unique_samples_seen_definition": "union of indices passed to training loader; scoring/evaluation excluded",
    }
    base = {**full_config(a), "schema_version": SCHEMA_VERSION, "key": key,
            "identity": identity, "config": full_config(a), "metadata": metadata,
            "attempt_id": uuid.uuid4().hex, "started_at": started_at}
    append_jsonl(a.results, {**base, "status": "started"})
    timings = {name: 0.0 for name in (
        "dense_scoring_time", "pruned_scoring_time", "selection_time", "train_time", "evaluation_time")}
    try:
        record = _run(a, data, verbose, device, timings, key)
        synchronize(device)
        record.update(base)
        record.update({k: round(v, 6) for k, v in timings.items()})
        record.update(status="completed", ended_at=utc_now(),
                      scoring_time=round(timings["dense_scoring_time"] + timings["pruned_scoring_time"], 6),
                      total_wall_time=round(time.perf_counter() - start, 6))
        # Legacy-compatible effective fraction; raw CLI settings are in config.
        record["frac"] = identity["frac"]
        append_jsonl(a.results, record)
        if verbose:
            print(f"  final {record['final_acc']:.2f}%; total {record['total_wall_time']:.2f}s -> {a.results}", flush=True)
        return record
    except BaseException as exc:
        failed = {**base, **{k: round(v, 6) for k, v in timings.items()},
                  "status": "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                  "ended_at": utc_now(), "error_type": type(exc).__name__, "error": str(exc),
                  "scoring_time": round(timings["dense_scoring_time"] + timings["pruned_scoring_time"], 6),
                  "total_wall_time": round(time.perf_counter() - start, 6)}
        try:
            append_jsonl(a.results, failed)
        except Exception as journal_error:
            warnings.warn(f"Could not append failure status: {journal_error}", RuntimeWarning)
        raise


def _run(a, data, verbose, device, timings, key):
    set_seed(a.seed)
    if data is None:
        data = load_cifar(a.dataset, a.data_root, device, a.limit_train)
    x_tr, y_tr, x_te, y_te = data

    model = load_dense(a, device)
    with timed(timings, "evaluation_time", device):
        dense_acc = evaluate(model, x_te, y_te, a.dataset, device)
    dense_loss = None
    if a.method in NEEDS_DENSE:
        with timed(timings, "dense_scoring_time", device):
            dense_loss, _, _ = per_sample_outputs(model, x_tr, y_tr, a.dataset, device)

    global_magnitude_prune(model, a.sparsity)
    sparsity = measure_sparsity(model)
    with timed(timings, "evaluation_time", device):
        pruned_acc = evaluate(model, x_te, y_te, a.dataset, device)

    idx, w, st = make_subset(a, model, data, dense_loss, device, timings)
    sel_stats = [st]
    seen = torch.zeros(len(y_tr), dtype=torch.bool, device=device)
    seen[idx] = True
    if verbose:
        print(f"[{key}] dense {dense_acc:.2f}% -> pruned {pruned_acc:.2f}% "
              f"(sparsity {sparsity:.3f}); subset {len(idx)} samples", flush=True)
        print(f"  weights: {st}", flush=True)

    loader = GpuLoader(x_tr, y_tr, a.dataset, a.batch_size, train=True, weights=w, indices=idx)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=a.wd)
    successful_updates = [0]
    def count_update(optimizer, args, kwargs):
        successful_updates[0] += 1
    opt.register_step_post_hook(count_update)
    total_steps = a.epochs * len(loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    curve, steps, samples_processed = [], 0, 0
    for ep in range(a.epochs):
        if a.reselect_every and ep > 0 and ep % a.reselect_every == 0:
            idx, w, st = make_subset(a, model, data, dense_loss, device, timings, epoch=ep)
            seen[idx] = True
            sel_stats.append(st)
            if verbose:
                print(f"  weights: {st}", flush=True)
            loader = GpuLoader(x_tr, y_tr, a.dataset, a.batch_size, train=True, weights=w, indices=idx)
        model.train()
        with timed(timings, "train_time", device):
            for xb, yb, wb, _ in loader:
                with autocast(device):
                    loss = (F.cross_entropy(model(xb), yb, reduction="none") * wb).mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                if steps < total_steps:
                    sched.step()
                steps += 1
                samples_processed += len(yb)
        with timed(timings, "evaluation_time", device):
            acc = evaluate(model, x_te, y_te, a.dataset, device)
        curve.append(round(acc, 2))
        if verbose and ((ep + 1) % 5 == 0 or ep + 1 == a.epochs):
            print(f"  epoch {ep + 1}/{a.epochs}  test acc {acc:.2f}%", flush=True)

    record = {
        "frac": 1.0 if a.method == "full" else a.frac,
        "n_selected": int(len(idx)), "measured_sparsity": round(sparsity, 4),
        "dense_acc": round(dense_acc, 2), "pruned_acc": round(pruned_acc, 2),
        "final_acc": curve[-1], "best_acc": max(curve), "curve": curve,
        "sel_stats": sel_stats, "steps": steps, "sgd_steps": successful_updates[0],
        "skipped_optimizer_steps": steps - successful_updates[0],
        "samples_processed": samples_processed, "unique_samples_seen": int(seen.sum().item()),
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
    }
    return record


if __name__ == "__main__":
    run(build_parser().parse_args())
