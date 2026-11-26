"""Cost-effectiveness table and figure from one audit (v2) journal.

    python scripts/cost_analysis.py results/raw/results_audit_t4_cost95_20.jsonl

Reads only completed v2 records, groups them by strategy (method, epochs, re-selection),
and writes results/cost_95_20.csv (mean and std over seeds) and results/cost_vs_acc.png.

Cost = scoring + selection + training time, measured on one GPU in one session.
Test-set evaluation is left out: it is measurement, not part of fine-tuning.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.audit import load_jsonl  # noqa: E402

TIMES = ["scoring_time", "selection_time", "train_time", "evaluation_time", "total_wall_time"]


def label(r):
    if r["method"] == "full":
        return f"full data, {r['epochs']} ep"
    name = f"{r['method']}, {round(r['frac'] * 100)}%"
    return name + (f", re-select every {r['reselect_every']} ep" if r["reselect_every"] else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("journal")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    recs = [r for r in load_jsonl(a.journal) if r.get("status") == "completed" and r.get("schema_version") == 2]
    if not recs:
        raise SystemExit("no completed v2 records")
    gpus = {r["metadata"]["gpu_model"] for r in recs}
    if len(gpus) != 1:
        raise SystemExit(f"records come from several GPUs, not comparable: {gpus}")
    df = pd.DataFrame([{**{k: r[k] for k in ["method", "frac", "epochs", "reselect_every", "seed",
                                               "final_acc", "sgd_steps", "unique_samples_seen"] + TIMES},
                        "strategy": label(r)} for r in recs])
    df["cost_time"] = df.scoring_time + df.selection_time + df.train_time

    cols = ["final_acc", "scoring_time", "selection_time", "train_time", "cost_time",
            "evaluation_time", "total_wall_time", "sgd_steps", "unique_samples_seen"]
    g = df.groupby("strategy")
    summ = g[cols].mean().round(2)
    summ.insert(1, "final_acc_std", g.final_acc.std().round(2))
    summ.insert(0, "seeds", g.size())
    summ = summ.sort_values("cost_time")
    summ.insert(len(summ.columns), "gpu", gpus.pop())
    os.makedirs(a.out, exist_ok=True)
    summ.to_csv(os.path.join(a.out, "cost_95_20.csv"))
    print(summ.to_string())

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, row in summ.iterrows():
        full = name.startswith("full")
        ax.errorbar(row.cost_time, row.final_acc, yerr=row.final_acc_std, fmt="s" if full else "o",
                    color="#1f5fa8" if full else "#777777", ms=7, capsize=3)
        ax.annotate(name, (row.cost_time, row.final_acc), xytext=(-10, 4) if full else (8, -14 if "re-select" in name else 6),
                    textcoords="offset points", fontsize=8, ha="right" if full else "left")
    ax.set_xlabel(f"fine-tuning cost: scoring + selection + training (s, {row.gpu})")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("95% sparsity, 20% data budget: accuracy vs cost (mean ± std, 3 seeds)", fontsize=10)
    ax.grid(alpha=.3)
    ax.margins(x=0.25, y=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "cost_vs_acc.png"), dpi=180)


if __name__ == "__main__":
    main()
