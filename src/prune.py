"""Global unstructured magnitude pruning via torch.nn.utils.prune.

The pruning masks stay attached during fine-tuning (weight = weight_orig * mask),
so pruned weights remain exactly zero while the rest are updated.
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def prunable_params(model, include_fc=True):
    params = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or (include_fc and isinstance(m, nn.Linear)):
            params.append((m, "weight"))
    return params


def global_magnitude_prune(model, sparsity, include_fc=True):
    """Zero out the `sparsity` fraction of conv/linear weights with the smallest |w|."""
    if sparsity <= 0:
        return model
    prune.global_unstructured(prunable_params(model, include_fc),
                              pruning_method=prune.L1Unstructured, amount=sparsity)
    return model


@torch.no_grad()
def measure_sparsity(model, include_fc=True):
    zeros = total = 0
    for m, name in prunable_params(model, include_fc):
        w = getattr(m, name)
        zeros += (w == 0).sum().item()
        total += w.numel()
    return zeros / max(total, 1)


def make_permanent(model):
    """Fold masks into weights (call before saving a final pruned checkpoint)."""
    for m, name in prunable_params(model):
        if prune.is_pruned(m):
            prune.remove(m, name)
    return model
