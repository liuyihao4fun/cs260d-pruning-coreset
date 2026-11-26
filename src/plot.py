"""Aggregate completed journals, keeping protocols and hardware separate."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .audit import IDENTITY_FIELDS, canonical_json, identity_key, is_completed, load_jsonl

UNKNOWN = "<unknown>"
GROUP_FIELDS = [k for k in IDENTITY_FIELDS if k != "seed"] + [
    "checkpoint_id", "source_sha256", "protocol", "schema_version", "gpu",
    "runtime_id", "timing_schema", "data_preloaded", "wall_clock_scope",
]
CONDITION_FIELDS = [k for k in GROUP_FIELDS if k not in ("sparsity", "method", "frac")]
METRICS = ["pruned_acc", "dense_acc", "train_time", "selection_time", "scoring_time",
           "dense_scoring_time", "pruned_scoring_time", "evaluation_time", "total_wall_time",
           "steps", "sgd_steps", "unique_samples_seen", "samples_processed"]
COLORS = {"random": "#7f7f7f", "el2n": "#ff7f0e", "craig": "#1f77b4",
          "damage_topk": "#9467bd", "damage_fl": "#d62728"}


def normalized_record(record):
    config = record.get("config", record)
    identity = record.get("identity", {})
    meta = record.get("metadata", {})
    if record.get("schema_version") == 2 and record.get("key") != identity_key(identity):
        raise ValueError("v2 result has an inconsistent identity/key")
    row = {k: identity.get(k, config.get(k, UNKNOWN)) for k in IDENTITY_FIELDS}
    if row["method"] == "full":
        row["frac"] = 1.0
    row.update(
        checkpoint_id=identity.get("checkpoint_sha256", meta.get("checkpoint_sha256",
                                   config.get("ckpt", UNKNOWN))),
        source_sha256=identity.get("source_sha256", UNKNOWN),
        protocol=identity.get("protocol", "legacy-unknown-protocol"),
        schema_version=record.get("schema_version", 1),
        gpu=meta.get("gpu_model") or record.get("gpu") or record.get("device", UNKNOWN),
        runtime_id=canonical_json({k: meta.get(k, UNKNOWN) for k in (
            "pytorch_version", "cuda_version", "cudnn_version", "device",
            "deterministic_algorithms", "cudnn_benchmark", "cudnn_deterministic",
            "cuda_matmul_allow_tf32", "cudnn_allow_tf32")}),
        timing_schema=meta.get("timing_schema", "legacy-selection-includes-pruned-scoring"),
        data_preloaded=meta.get("data_preloaded", UNKNOWN),
        wall_clock_scope=meta.get("wall_clock_scope", UNKNOWN),
    )
    row = {k: "<none>" if v is None else v for k, v in row.items()}
    return {**row, **{k: record.get(k, float("nan")) for k in METRICS},
            "final_acc": record["final_acc"], "best_acc": record.get("best_acc", float("nan"))}


def aggregate_records(records, metric="final_acc"):
    rows, seen = [], {}
    for record in records:
        if not is_completed(record):
            continue
        row = normalized_record(record)
        if row["seed"] == UNKNOWN:
            raise ValueError("cannot aggregate a completed result without its seed")
        token = tuple(row[k] for k in GROUP_FIELDS) + (row["seed"],)
        encoded = canonical_json(record)
        if token in seen:
            if seen[token] != encoded:
                raise ValueError("conflicting completed results for the same configuration/seed; use distinct tags for intentional repeats")
            continue
        seen[token] = encoded
        rows.append(row)
    if not rows:
        raise ValueError("no completed results found")
    if any(r["schema_version"] == 1 for r in rows):
        warnings.warn("legacy results: missing configuration/provenance stays <unknown>; legacy timing is not v2 timing", RuntimeWarning)
    df = pd.DataFrame(rows)
    if df[metric].isna().any():
        raise ValueError(f"some completed results lack {metric}")
    aggregations = {"acc_mean": (metric, "mean"), "acc_std": (metric, "std"),
                    "n_seeds": ("seed", "nunique")}
    aggregations.update({k: (k, "mean") for k in METRICS})
    summary = df.groupby(GROUP_FIELDS, dropna=False, sort=False).agg(**aggregations).reset_index()
    front = ["dataset", "sparsity", "method", "frac", "epochs", "tag",
             "acc_mean", "acc_std", "n_seeds", "pruned_acc", "dense_acc",
             "train_time", "selection_time"]
    return summary[front + [c for c in summary.columns if c not in front]]


def figure_groups(summary):
    return list(summary.groupby(CONDITION_FIELDS, dropna=False, sort=False))


def write_outputs(summary, out, overwrite=False):
    out = Path(out)
    groups = figure_groups(summary)
    plans = []
    for condition, frame in groups:
        spec = dict(zip(CONDITION_FIELDS, condition))
        # Pandas may return numpy scalar keys, which need native JSON scalars.
        spec = {k: v.item() if hasattr(v, "item") else v for k, v in spec.items()}
        suffix = hashlib.sha256(canonical_json(spec).encode()).hexdigest()[:12]
        name = "acc_vs_frac.png" if len(groups) == 1 else f"acc_vs_frac_{suffix}.png"
        plans.append((out / name, spec, frame))
    targets = [out / "summary.csv", out / "figure_manifest.json"] + [p[0] for p in plans]
    if not overwrite and any(p.exists() for p in targets):
        raise FileExistsError("output already exists; choose a new --out directory (or explicitly --overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    manifest = []
    for path, spec, frame in plans:
        sparsities = sorted(frame.sparsity.unique())
        fig, axes = plt.subplots(1, len(sparsities), figsize=(4.5 * len(sparsities), 3.8), squeeze=False)
        for ax, sp in zip(axes[0], sparsities):
            part = frame[frame.sparsity == sp]
            for method, sm in part.groupby("method", sort=False):
                if method == "full":
                    if len(sm) != 1:
                        raise ValueError("ambiguous full-data reference")
                    ax.axhline(sm.acc_mean.iloc[0], ls="--", c="k", label="full data (same configuration)")
                else:
                    sm = sm.sort_values("frac")
                    ax.errorbar(sm.frac * 100, sm.acc_mean, yerr=sm.acc_std.fillna(0),
                                marker="o", capsize=3, label=method, color=COLORS.get(method))
            ax.set_title(f"sparsity {sp:.0%}")
            ax.set_xlabel("fine-tuning data (%)")
            ax.set_ylabel("test accuracy (%)")
            ax.grid(alpha=.3)
            ax.legend(fontsize=7)
        fig.suptitle(f"epochs={spec['epochs']}, rs={spec['reselect_every']}, uniform={spec['no_weights']}, GPU={spec['gpu']}", fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        manifest.append({"file": path.name, "configuration": spec})
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2))
    return targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", default=["results/results.jsonl"])
    p.add_argument("--out", default="results/audit_figures")
    p.add_argument("--metric", default="final_acc", choices=["final_acc", "best_acc"])
    p.add_argument("--overwrite", action="store_true", help="explicitly replace generated outputs")
    a = p.parse_args()
    summary = aggregate_records([r for f in a.results for r in load_jsonl(f)], a.metric)
    for path in write_outputs(summary, a.out, a.overwrite):
        print(path)


if __name__ == "__main__":
    main()
