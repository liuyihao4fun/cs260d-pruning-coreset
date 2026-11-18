"""Subset selection methods for post-pruning fine-tuning.

Every method has the signature  fn(ctx, frac) -> (indices, weights)
  indices : LongTensor of selected training-set indices
  weights : FloatTensor of per-sample loss weights (mean 1), same length

`ctx` is a SelectionContext holding everything computed once before selection:
labels, and per-sample outputs of the pruned model (and of the dense model).
All methods are class-balanced: each class gets round(frac * n_class) samples.

To add a method: write fn(ctx, frac) in a new file and register it in METHODS.
"""
from dataclasses import dataclass
from typing import Optional

import torch

from .random_sel import select_random
from .score_based import select_el2n, select_damage_topk
from .craig import select_craig, select_damage_fl


@dataclass
class SelectionContext:
    y: torch.Tensor                 # labels [N]
    num_classes: int
    loss: torch.Tensor              # per-sample loss of the PRUNED model [N]
    probs: torch.Tensor             # softmax of the pruned model [N, C]
    feats: torch.Tensor             # penultimate features of the pruned model [N, D]
    dense_loss: Optional[torch.Tensor] = None  # per-sample loss of the DENSE model [N]
    damage_alpha: float = 1.0       # see craig.select_damage_fl
    generator: Optional[torch.Generator] = None


METHODS = {
    "random": select_random,
    "el2n": select_el2n,
    "craig": select_craig,
    "damage_topk": select_damage_topk,
    "damage_fl": select_damage_fl,
}

NEEDS_DENSE = {"damage_topk", "damage_fl"}


def select(method, ctx, frac):
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {list(METHODS)}")
    idx, w = METHODS[method](ctx, frac)
    w = w.float()
    return idx, w * (len(w) / w.sum())   # normalise weights to mean 1
