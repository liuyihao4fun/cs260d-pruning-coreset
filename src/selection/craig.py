"""CRAIG (Mirzasoleiman et al., ICML 2020) and the damage-weighted variant.

CRAIG picks a subset S whose weighted gradient sum approximates the full gradient
by greedily maximising the facility-location function

    F(S) = sum_i  w_i * max_{j in S} sim(i, j),     sim(i, j) = d_max - ||g_i - g_j||

where g_i is a cheap proxy for sample i's gradient. Each selected j then gets a
training weight gamma_j = number of samples i whose closest element in S is j.

Gradient proxy: the exact gradient of the loss w.r.t. the last linear layer,
    g_i = (p_i - y_i) (outer) [h_i, 1]
with p_i the softmax output and h_i the penultimate features. Pairwise distances
are computed without materialising the (C x (D+1))-dim vectors, using
    <a(x)b, c(x)d> = <a, c> <b, d>.

Plain CRAIG uses w_i = 1. `select_damage_fl` (our method) uses
    w_i = alpha + damage_i / mean(damage)
so samples that pruning hurt the most count more in the coverage objective,
while alpha > 0 keeps every sample represented (alpha -> inf recovers CRAIG).
"""
import torch
import torch.nn.functional as F

from .random_sel import per_class_budget
from .score_based import damage_scores


def _grad_distances(err, h):
    """Pairwise ||g_i - g_j|| for g = err (outer) [h, 1]."""
    h1 = torch.cat([h, torch.ones(len(h), 1, device=h.device)], dim=1)
    ee = err @ err.T
    hh = h1 @ h1.T
    dot = ee * hh                              # <g_i, g_j>
    sq = dot.diagonal()
    d2 = sq[:, None] + sq[None, :] - 2 * dot
    return d2.clamp(min=0).sqrt()


def facility_location_greedy(sim, k, w=None):
    """Greedy maximisation of sum_i w_i max_{j in S} sim[i, j].

    sim: [n, n] non-negative similarities (row = sample to cover, col = candidate).
    Returns (selected column indices, cluster sizes gamma_j).
    """
    n = sim.shape[0]
    w = torch.ones(n, device=sim.device) if w is None else w
    cur = torch.zeros(n, device=sim.device)         # current coverage of each row
    selected = []
    available = torch.ones(n, dtype=torch.bool, device=sim.device)
    for _ in range(min(k, n)):
        gains = (w[:, None] * (sim - cur[:, None]).clamp(min=0)).sum(0)
        gains[~available] = -1
        j = int(gains.argmax())
        selected.append(j)
        available[j] = False
        cur = torch.maximum(cur, sim[:, j])
    sel = torch.tensor(selected, device=sim.device)
    assign = sim[:, sel].argmax(1)                   # nearest selected element
    gamma = torch.bincount(assign, minlength=len(sel)).float()
    return sel, gamma


def _fl_select(ctx, frac, point_weights=None):
    onehot = F.one_hot(ctx.y, ctx.num_classes).float()
    err = ctx.probs.float() - onehot
    chosen, weights = [], []
    for c in range(ctx.num_classes):
        cls_idx = (ctx.y == c).nonzero(as_tuple=True)[0]
        if len(cls_idx) == 0:
            continue
        k = per_class_budget(len(cls_idx), frac)
        dist = _grad_distances(err[cls_idx], ctx.feats[cls_idx].float())
        sim = dist.max() - dist
        w = None if point_weights is None else point_weights[cls_idx]
        sel, gamma = facility_location_greedy(sim, k, w)
        chosen.append(cls_idx[sel])
        weights.append(gamma.clamp(min=1))
    return torch.cat(chosen), torch.cat(weights)


def select_craig(ctx, frac):
    return _fl_select(ctx, frac)


def select_damage_fl(ctx, frac):
    d = damage_scores(ctx)
    w = ctx.damage_alpha + d / d.mean().clamp(min=1e-8)
    return _fl_select(ctx, frac, point_weights=w)
