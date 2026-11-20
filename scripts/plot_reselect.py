"""Figures for the re-selection experiment (subset re-chosen every 5 epochs).

    python scripts/plot_reselect.py

Reads results/summary_all.csv (one-shot, 30 epochs), results/reselect_runs.csv
(per-seed re-selection runs) and results/reselect_curves.csv (example test-acc curves);
writes results/reselect_vs_oneshot.png, results/reselect_curves.png and
results/reselect_summary.csv.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

C = {"random": "#7f7f7f", "craig": "#1f77b4", "damage_fl": "#d62728"}
M = ["random", "craig", "damage_fl"]

one = pd.read_csv("results/summary_all.csv")
one = one[(one.epochs == 30) & one.method.isin(M)][["sparsity", "method", "frac", "mean", "std"]]
rs = pd.read_csv("results/reselect_runs.csv")
rss = (rs.groupby(["sparsity", "method", "frac"])
         .agg(mean=("final_acc", "mean"), median=("final_acc", "median"), std=("final_acc", "std"),
              best_mean=("best_acc", "mean"), collapsed=("min_acc_after_ep5", lambda x: int((x < 50).sum())))
         .round(2).reset_index())
summ = one.merge(rss, on=["sparsity", "method", "frac"], suffixes=("_oneshot", "_rs5"))
summ.to_csv("results/reselect_summary.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
for ax, sp in zip(axes, [0.9, 0.95, 0.98]):
    for m in M:
        o = one[(one.sparsity == sp) & (one.method == m)].sort_values("frac")
        r = rss[(rss.sparsity == sp) & (rss.method == m)].sort_values("frac")
        ax.plot(o.frac * 100, o["mean"], ls="--", marker="o", color=C[m], alpha=.6, lw=1.2)
        ax.plot(r.frac * 100, r["median"], ls="-", marker="s", color=C[m], lw=2, label=m)
    lo = one[one.sparsity == sp]["mean"].min() - 1.5
    ax.set_ylim(lo, one[one.sparsity == sp]["mean"].max() + 2)
    for _, r in rss[rss.sparsity == sp].iterrows():  # medians that fall off the axis
        if r["median"] < lo:
            ax.annotate(f"{r['median']:.1f}", (r.frac * 100, lo), xytext=(0, 14), textcoords="offset points",
                        ha="center", color=C[r.method], fontsize=8,
                        arrowprops=dict(arrowstyle="->", color=C[r.method]))
    ax.set_title(f"sparsity {sp:.0%}")
    ax.set_xticks([5, 10, 20]); ax.set_xlabel("fine-tuning data (%)"); ax.grid(alpha=.3)
axes[0].set_ylabel("test acc (%)"); axes[0].legend(fontsize=8)
fig.suptitle("Solid squares: re-select every 5 epochs (median of 3 seeds). Dashed circles: one-shot (mean).", fontsize=10)
fig.tight_layout(); fig.savefig("results/reselect_vs_oneshot.png", dpi=180)

cv = pd.read_csv("results/reselect_curves.csv")
fig, ax = plt.subplots(figsize=(7, 4))
for _, r in cv.iterrows():
    y = [float(v) for v in r.curve.split()]
    ax.plot(range(1, len(y) + 1), y, color=C[r.method], lw=2, label=r.method)
for e in range(5, 30, 5):
    ax.axvline(e + 0.5, c="k", lw=.6, ls=":")
ax.set_xlabel("epoch (dotted = re-selection)"); ax.set_ylabel("test acc (%)")
ax.set_title("95% sparsity, 20% data, seed 0, re-selecting every 5 epochs", fontsize=10)
ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig("results/reselect_curves.png", dpi=180)
print(summ.to_string(index=False))
