"""Score-based selection: keep the top-scoring samples within each class."""
import torch
import torch.nn.functional as F

from .random_sel import per_class_budget


def _topk_per_class(ctx, scores, frac):
    chosen = []
    for c in range(ctx.num_classes):
        cls_idx = (ctx.y == c).nonzero(as_tuple=True)[0]
        if len(cls_idx) == 0:
            continue
        k = per_class_budget(len(cls_idx), frac)
        top = scores[cls_idx].topk(min(k, len(cls_idx))).indices
        chosen.append(cls_idx[top])
    idx = torch.cat(chosen)
    return idx, torch.ones(len(idx), device=idx.device)


def el2n_scores(ctx):
    """EL2N = || softmax(f(x)) - onehot(y) ||_2, computed with the pruned model."""
    onehot = F.one_hot(ctx.y, ctx.num_classes).float()
    return (ctx.probs - onehot).norm(dim=1)


def damage_scores(ctx):
    """How much pruning hurt each sample: loss_pruned - loss_dense (clipped at 0)."""
    if ctx.dense_loss is None:
        raise ValueError("damage scores need the dense model's per-sample loss")
    return (ctx.loss - ctx.dense_loss).clamp(min=0)


def select_el2n(ctx, frac):
    """Hardest samples for the pruned model."""
    return _topk_per_class(ctx, el2n_scores(ctx), frac)


def select_damage_topk(ctx, frac):
    """Samples whose loss increased the most because of pruning (no diversity term)."""
    return _topk_per_class(ctx, damage_scores(ctx), frac)
