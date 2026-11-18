import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from .data import GpuLoader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def autocast(device):
    """fp16 autocast on CUDA (works on T4/P100/A100); no-op elsewhere."""
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def evaluate(model, x, y, dataset, device, batch_size=1000):
    model.eval()
    correct = 0
    for xb, yb, _, _ in GpuLoader(x, y, dataset, batch_size, train=False):
        with autocast(device):
            correct += (model(xb).argmax(1) == yb).sum().item()
    return 100.0 * correct / len(x)


@torch.no_grad()
def per_sample_outputs(model, x, y, dataset, device, batch_size=1000):
    """One pass over the full training set without augmentation.

    Returns per-sample loss, softmax probabilities and penultimate features,
    which the selection methods use as scores / gradient proxies.
    """
    model.eval()
    losses, probs, feats = [], [], []
    for xb, yb, _, _ in GpuLoader(x, y, dataset, batch_size, train=False):
        with autocast(device):
            logits, h = model(xb, return_features=True)
        logits = logits.float()
        losses.append(F.cross_entropy(logits, yb, reduction="none"))
        probs.append(logits.softmax(1))
        feats.append(h.float())
    return torch.cat(losses), torch.cat(probs), torch.cat(feats)


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
