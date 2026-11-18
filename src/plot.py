"""Aggregate results.jsonl into a summary table and the main figure.

    python -m src.plot --results results/results.jsonl --out results/

Writes summary.csv (mean/std over seeds) and acc_vs_frac.png
(one panel per sparsity; x = data fraction, y = accuracy after fine-tuning;
dashed line = fine-tuning on 100% of the data).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .utils import load_jsonl

COLORS = {"random": "#7f7f7f", "el2n": "#ff7f0e", "craig": "#1f77b4",
          "damage_topk": "#9467bd", "damage_fl": "#d62728"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", default=["results/results.jsonl"],
                   help="one or more jsonl files (e.g. one per teammate)")
    p.add_argument("--out", default="results/")
    p.add_argument("--metric", default="final_acc", choices=["final_acc", "best_acc"])
    a = p.parse_args()

    rows = [r for f in a.results for r in load_jsonl(f)]
    if not rows:
        raise SystemExit("no results found")
    df = pd.DataFrame(rows).drop_duplicates("key")
    group = ["dataset", "sparsity", "method", "frac", "epochs", "tag"]
    summ = (df.groupby(group)
              .agg(acc_mean=(a.metric, "mean"), acc_std=(a.metric, "std"), n_seeds=("seed", "count"),
                   pruned_acc=("pruned_acc", "mean"), dense_acc=("dense_acc", "mean"),
                   train_time=("train_time", "mean"), selection_time=("selection_time", "mean"))
              .reset_index().round(2))
    os.makedirs(a.out, exist_ok=True)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    print(summ.to_string(index=False))

    sps = sorted(summ.sparsity.unique())
    fig, axes = plt.subplots(1, len(sps), figsize=(4.5 * len(sps), 3.6), squeeze=False)
    for ax, sp in zip(axes[0], sps):
        s = summ[summ.sparsity == sp]
        full = s[s.method == "full"]
        if len(full):
            ax.axhline(full.acc_mean.iloc[0], ls="--", c="k", lw=1, label="full data")
        for m, sm in s[s.method != "full"].groupby("method"):
            sm = sm.sort_values("frac")
            ax.errorbar(sm.frac * 100, sm.acc_mean, yerr=sm.acc_std.fillna(0), marker="o",
                        capsize=3, label=m, color=COLORS.get(m))
        ax.set_title(f"sparsity {sp:.0%}  (no FT: {s.pruned_acc.iloc[0]:.1f}%)")
        ax.set_xlabel("fine-tuning data (%)")
        ax.set_ylabel("test accuracy (%)")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(a.out, "acc_vs_frac.png")
    fig.savefig(path, dpi=200)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
