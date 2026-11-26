"""Train the dense (unpruned) ResNet-18 that every experiment starts from.

    python -m src.train_dense --dataset cifar10 --epochs 40 --out checkpoints/resnet18_cifar10.pt

~40 epochs reaches roughly 94-95% on CIFAR-10. Do this ONCE and save the
checkpoint to Google Drive; subsequent runs reuse the same file.
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F

from .data import GpuLoader, NUM_CLASSES, load_cifar
from .models import resnet18
from .utils import autocast, evaluate, get_device, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10", choices=list(NUM_CLASSES))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--limit_train", type=int, default=None, help="debug: use first N images")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    set_seed(a.seed)
    device = get_device()
    x_tr, y_tr, x_te, y_te = load_cifar(a.dataset, a.data_root, device, a.limit_train)
    model = resnet18(NUM_CLASSES[a.dataset]).to(device)
    loader = GpuLoader(x_tr, y_tr, a.dataset, a.batch_size, train=True, drop_last=True)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, nesterov=True, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs,
                                                steps_per_epoch=len(loader), pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        for xb, yb, _, _ in loader:
            with autocast(device):
                loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        acc = evaluate(model, x_te, y_te, a.dataset, device)
        print(f"epoch {ep + 1}/{a.epochs}  loss {loss.item():.3f}  test acc {acc:.2f}%  ({time.time() - t0:.1f}s)",
              flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "dataset": a.dataset, "test_acc": acc}, a.out)
    print(f"saved {a.out}  (test acc {acc:.2f}%)")


if __name__ == "__main__":
    main()
