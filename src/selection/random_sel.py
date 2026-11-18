import torch


def per_class_budget(n_c, frac):
    return max(1, int(round(frac * n_c)))


def select_random(ctx, frac):
    """Class-balanced uniform random subset (the key baseline)."""
    chosen = []
    for c in range(ctx.num_classes):
        cls_idx = (ctx.y == c).nonzero(as_tuple=True)[0]
        if len(cls_idx) == 0:
            continue
        k = per_class_budget(len(cls_idx), frac)
        perm = torch.randperm(len(cls_idx), generator=ctx.generator, device="cpu")
        chosen.append(cls_idx[perm[:k].to(cls_idx.device)])
    idx = torch.cat(chosen)
    return idx, torch.ones(len(idx), device=idx.device)
