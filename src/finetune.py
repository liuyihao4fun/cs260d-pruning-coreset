"""Prune a dense checkpoint, select a training subset, fine-tune, evaluate.

Single run:
    python -m src.finetune --ckpt checkpoints/resnet18_cifar10.pt \
        --sparsity 0.9 --method craig --frac 0.1 --seed 0

Each run appends one JSON line to --results (default results/results.jsonl).
Many runs: see src/run_grid.py.
"""
import argparse
import copy
import time

import torch
import torch.nn.functional as F

from .data import GpuLoader, NUM_CLASSES, load_cifar
from .models import resnet18
from .prune import global_magnitude_prune, measure_sparsity
from .selection import METHODS, NEEDS_DENSE, SelectionContext, select
from .utils import (append_jsonl, autocast, evaluate, get_device, per_sample_outputs,
                    set_seed)

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


def run_key(a):
    frac = 1.0 if a.method == "full" else a.frac
    key = f"{a.dataset}_sp{a.sparsity}_{a.method}_fr{frac}_ep{a.epochs}_seed{a.seed}"
    if a.reselect_every:
        key += f"_rs{a.reselect_every}"
    if a.no_weights:
        key += "_nw"
    if a.method == "damage_fl" and a.damage_alpha != 1.0:
        key += f"_a{a.damage_alpha}"
    return key + (f"_{a.tag}" if a.tag else "")


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


def make_subset(a, model, data, dense_loss, device, epoch=0):
    """Returns (indices, weights, seconds spent selecting, weight diagnostics).

    `epoch` offsets the random generator so that re-selection (random method) draws a
    new subset each time; epoch 0 keeps the original seed, so one-shot runs are unchanged.
    """
    x_tr, y_tr = data[0], data[1]
    t0 = time.time()
    if a.method == "full":
        idx = torch.arange(len(y_tr), device=device)
        return idx, torch.ones(len(idx), device=device), 0.0, {}
    loss, probs, feats = per_sample_outputs(model, x_tr, y_tr, a.dataset, device)
    ctx = SelectionContext(y=y_tr, num_classes=NUM_CLASSES[a.dataset], loss=loss, probs=probs,
                           feats=feats, dense_loss=dense_loss, damage_alpha=a.damage_alpha,
                           generator=torch.Generator().manual_seed(a.seed + 1000 * epoch))
    idx, w = select(a.method, ctx, a.frac)
    if a.no_weights:
        w = torch.ones_like(w)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    return idx, w, dt, weight_stats(w, probs, y_tr, epoch)


def run(a, data=None, verbose=True):
    set_seed(a.seed)
    device = get_device()
    if data is None:
        data = load_cifar(a.dataset, a.data_root, device, a.limit_train)
    x_tr, y_tr, x_te, y_te = data

    model = load_dense(a, device)
    dense_acc = evaluate(model, x_te, y_te, a.dataset, device)
    dense_loss = None
    if a.method in NEEDS_DENSE:
        dense_loss, _, _ = per_sample_outputs(model, x_tr, y_tr, a.dataset, device)

    global_magnitude_prune(model, a.sparsity)
    sparsity = measure_sparsity(model)
    pruned_acc = evaluate(model, x_te, y_te, a.dataset, device)

    idx, w, sel_time, st = make_subset(a, model, data, dense_loss, device)
    sel_stats = [st]
    if verbose:
        print(f"[{run_key(a)}] dense {dense_acc:.2f}% -> pruned {pruned_acc:.2f}% "
              f"(sparsity {sparsity:.3f}); subset {len(idx)} samples, selection {sel_time:.1f}s", flush=True)
        print(f"  weights: {st}", flush=True)

    loader = GpuLoader(x_tr, y_tr, a.dataset, a.batch_size, train=True, weights=w, indices=idx)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=a.wd)
    total_steps = a.epochs * len(loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    curve, train_time, steps = [], 0.0, 0
    for ep in range(a.epochs):
        if a.reselect_every and ep > 0 and ep % a.reselect_every == 0:
            idx, w, t, st = make_subset(a, model, data, dense_loss, device, epoch=ep)
            sel_time += t
            sel_stats.append(st)
            if verbose:
                print(f"  weights: {st}", flush=True)
            loader = GpuLoader(x_tr, y_tr, a.dataset, a.batch_size, train=True, weights=w, indices=idx)
        model.train()
        t0 = time.time()
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
        if device == "cuda":
            torch.cuda.synchronize()
        train_time += time.time() - t0
        acc = evaluate(model, x_te, y_te, a.dataset, device)
        curve.append(round(acc, 2))
        if verbose and ((ep + 1) % 5 == 0 or ep + 1 == a.epochs):
            print(f"  epoch {ep + 1}/{a.epochs}  test acc {acc:.2f}%", flush=True)

    record = {
        "key": run_key(a), **{k: v for k, v in vars(a).items() if k not in ("data_root",)},
        "frac": 1.0 if a.method == "full" else a.frac,
        "n_selected": int(len(idx)), "measured_sparsity": round(sparsity, 4),
        "dense_acc": round(dense_acc, 2), "pruned_acc": round(pruned_acc, 2),
        "final_acc": curve[-1], "best_acc": max(curve), "curve": curve,
        "sel_stats": sel_stats, "steps": steps, "selection_time": round(sel_time, 2), "train_time": round(train_time, 2),
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
    }
    append_jsonl(a.results, record)
    if verbose:
        print(f"  final {curve[-1]:.2f}%  (train {train_time:.0f}s, select {sel_time:.0f}s) -> {a.results}",
              flush=True)
    return record


if __name__ == "__main__":
    run(build_parser().parse_args())
