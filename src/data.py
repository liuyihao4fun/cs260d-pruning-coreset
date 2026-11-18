"""CIFAR data held entirely on the GPU, with batched GPU augmentation.

Colab/Kaggle machines have only 2 CPU cores, so a normal torchvision DataLoader
becomes the bottleneck for ResNet-18 on CIFAR. Keeping the whole dataset on the
GPU (~150 MB as uint8) and augmenting there makes each epoch several times faster.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def load_cifar(name, root="./data", device="cuda", limit_train=None):
    """Returns (x_train, y_train, x_test, y_test) as tensors on `device`.

    Images are uint8 NCHW; normalisation happens per batch in `GpuLoader`.
    `limit_train` keeps only the first N training images (for quick debugging).
    """
    cls = {"cifar10": torchvision.datasets.CIFAR10,
           "cifar100": torchvision.datasets.CIFAR100}[name]
    tr = cls(root, train=True, download=True)
    te = cls(root, train=False, download=True)

    def to_t(ds, limit=None):
        x = torch.from_numpy(np.asarray(ds.data)).permute(0, 3, 1, 2).contiguous()
        y = torch.tensor(ds.targets, dtype=torch.long)
        if limit:
            x, y = x[:limit], y[:limit]
        return x.to(device), y.to(device)

    x_tr, y_tr = to_t(tr, limit_train)
    x_te, y_te = to_t(te, limit_train // 5 if limit_train else None)
    return x_tr, y_tr, x_te, y_te


class GpuLoader:
    """Minimal DataLoader replacement for tensors that already live on the GPU.

    Yields (x, y, w, idx): normalised float images, labels, per-sample loss
    weights (all ones unless given, e.g. CRAIG weights) and dataset indices.
    """

    def __init__(self, x, y, dataset, batch_size=256, train=True, weights=None,
                 indices=None, drop_last=False):
        self.x, self.y = x, y
        self.batch_size, self.train, self.drop_last = batch_size, train, drop_last
        dev = x.device
        self.indices = (torch.arange(len(x), device=dev) if indices is None
                        else torch.as_tensor(indices, device=dev, dtype=torch.long))
        self.weights = (torch.ones(len(self.indices), device=dev) if weights is None
                        else torch.as_tensor(weights, device=dev, dtype=torch.float))
        mean, std = STATS[dataset]
        self.mean = torch.tensor(mean, device=dev).view(1, 3, 1, 1) * 255
        self.std = torch.tensor(std, device=dev).view(1, 3, 1, 1) * 255

    def __len__(self):
        n = len(self.indices)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def _augment(self, x):
        # random crop 32x32 from 4-pixel zero padding, per sample
        b = x.shape[0]
        xp = F.pad(x, (4, 4, 4, 4))
        i = torch.randint(0, 9, (b,), device=x.device)
        j = torch.randint(0, 9, (b,), device=x.device)
        ar = torch.arange(32, device=x.device)
        rows = (i[:, None] + ar)[:, None, :, None]
        cols = (j[:, None] + ar)[:, None, None, :]
        x = xp[torch.arange(b, device=x.device)[:, None, None, None],
               torch.arange(3, device=x.device)[None, :, None, None], rows, cols]
        # random horizontal flip
        flip = torch.rand(b, device=x.device) < 0.5
        return torch.where(flip[:, None, None, None], x.flip(3), x)

    def __iter__(self):
        n = len(self.indices)
        order = torch.randperm(n, device=self.x.device) if self.train else torch.arange(n, device=self.x.device)
        for s in range(0, n, self.batch_size):
            pos = order[s:s + self.batch_size]
            if self.drop_last and len(pos) < self.batch_size:
                break
            idx = self.indices[pos]
            xb = self.x[idx].float()
            if self.train:
                xb = self._augment(xb)
            xb = (xb - self.mean) / self.std
            yield xb, self.y[idx], self.weights[pos], idx
