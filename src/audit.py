"""Versioned experiment identity and append-only, single-writer result journals.

This module deliberately has no ML dependencies, so recovery can be checked on CPU.
Legacy records remain readable, but never satisfy a v2 resume check.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import warnings
from datetime import datetime, timezone

SCHEMA_VERSION = 2
PROTOCOL = "experiment-audit-v2"
IDENTITY_FIELDS = (
    "dataset", "sparsity", "method", "frac", "epochs", "lr", "batch_size", "wd",
    "seed", "reselect_every", "no_weights", "damage_alpha", "limit_train", "tag",
)
TRAINING_FILES = (
    "audit.py", "data.py", "finetune.py", "models.py", "prune.py", "utils.py",
    "selection/__init__.py", "selection/random_sel.py", "selection/score_based.py",
    "selection/craig.py",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in TRAINING_FILES:
        digest.update(name.encode() + b"\0" + (root / name).read_bytes() + b"\0")
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity_key(identity):
    return "v2_" + hashlib.sha256(canonical_json(identity).encode()).hexdigest()


def experiment_identity(a, checkpoint_sha256=None, source_hash=None):
    identity = {name: getattr(a, name) for name in IDENTITY_FIELDS}
    if a.method == "full":
        identity["frac"] = 1.0
    identity.update(
        protocol=PROTOCOL,
        checkpoint_sha256=checkpoint_sha256 or file_sha256(a.ckpt),
        source_sha256=source_hash or source_sha256(),
    )
    return identity


def run_key(a):
    return identity_key(experiment_identity(a))


def full_config(a):
    return {k: v for k, v in vars(a).items() if not k.startswith("_")}


def git_metadata():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL,
                text=True, timeout=10,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status),
            "status_porcelain": status}


def is_completed(record):
    status = record.get("status")
    value = record.get("final_acc")
    completed = status == "completed" or (status is None and record.get("schema_version") != 2)
    return (completed and isinstance(value, (int, float))
            and not isinstance(value, bool) and math.isfinite(value))


def completed_keys(records):
    """Only explicit, internally consistent v2 completions can skip new work."""
    return {r["key"] for r in records
            if r.get("schema_version") == SCHEMA_VERSION
            and r.get("status") == "completed" and is_completed(r)
            and isinstance(r.get("identity"), dict)
            and r["identity"].get("protocol") == PROTOCOL
            and r.get("key") == identity_key(r["identity"])}


def _decode_records(raw, path):
    lines = raw.splitlines()
    records = []
    last = max((i for i, line in enumerate(lines) if line.strip()), default=-1)
    damaged_tail = None
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            # An earlier append may have preserved an interrupted fragment and
            # explicitly marked it. Never ignore arbitrary interior corruption.
            marker = None
            following = next((v for v in lines[i + 1:] if v.strip()), None)
            if following is not None:
                try:
                    marker = json.loads(following)
                except (ValueError, UnicodeDecodeError):
                    pass
            expected = hashlib.sha256(line).hexdigest()
            if (isinstance(marker, dict) and marker.get("status") == "journal_recovery"
                    and marker.get("damaged_line_sha256") == expected):
                warnings.warn(f"{path}:{i + 1}: preserved interrupted write", RuntimeWarning)
                continue
            if i == last:
                damaged_tail = line
                warnings.warn(f"{path}:{i + 1}: ignoring incomplete JSONL tail", RuntimeWarning)
                continue
            raise ValueError(f"{path}:{i + 1}: corrupt interior JSONL record") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{i + 1}: JSONL record must be an object")
        records.append(record)
    return records, damaged_tail


def load_jsonl(path):
    p = Path(path)
    return _decode_records(p.read_bytes(), p)[0] if p.exists() else []


def append_jsonl(path, record):
    """Append + fsync without truncating bytes, even after a partial final write.

    Use one writer per file (including across Colab accounts). POSIX locks provide
    local process protection; Google Drive is not a distributed locking service.
    """
    import fcntl
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(record).encode() + b"\n"
    with p.open("a+b") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read()
            _, tail = _decode_records(raw, p)
            if raw and not raw.endswith(b"\n"):
                f.write(b"\n")
            if tail is not None:
                marker = {"status": "journal_recovery", "timestamp": utc_now(),
                          "damaged_line_sha256": hashlib.sha256(tail).hexdigest()}
                f.write(canonical_json(marker).encode() + b"\n")
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
