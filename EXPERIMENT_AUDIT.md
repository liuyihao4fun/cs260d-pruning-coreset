# Experiment audit v2

## Scope and compatibility

Training/model/pruning/selection mathematics and default hyperparameters are unchanged. Existing `results/*.csv` and PNGs are frozen historical exports. The original notebook is archived byte-for-byte as `notebooks/colab_run_legacy.ipynb`; the active notebook is regenerated with the current source.

Identity is SHA256 over canonical JSON containing dataset, sparsity, method, effective fraction, epochs, LR, batch size, weight decay, seed, re-selection, uniform-weight flag, damage alpha, training limit, tag, checkpoint **content** SHA256, training-source SHA256 and protocol version. The source fingerprint covers the training and audit modules, not plotting/docs/Notebook. A source change conservatively creates new identities. Checkpoint relocation does not change identity. Full CLI configuration, original checkpoint path, Git commit/dirty state and runtime information are retained separately.

The cache location (`data_root`) and result destination do not define scientific identity. This assumes the official torchvision CIFAR files/ordering; custom modified datasets are outside this protocol. Runtime/GPU metadata is retained and aggregation separates runtime profiles. Identity represents the configured experiment, not a guarantee of bitwise reproducibility across hardware. No deterministic flag/default has been changed.

Legacy JSONL dictionaries with finite final_acc remain readable. Missing settings/provenance stay unknown; do not impute historical defaults. Legacy completions never satisfy v2 resume checks. Conflicting completions for one configuration/seed are errors; exact duplicate copies are deduplicated. Intentional repeats require distinct tags. Historical accuracy is not rewritten or reinterpreted as newly measured data.

## Journal and restart semantics

Each run appends started, then completed/failed/interrupted with a common attempt_id, full config, identity and metadata. Abrupt process death can leave started only. Only a verified v2 completed record with finite final_acc skips work, for both single-run and Grid entry points. Incomplete runs restart from the dense checkpoint; epoch-level optimizer/model checkpointing was deliberately not added.

One writer per journal is required. Each teammate/runtime/shard uses a separate Drive JSONL. Appends are flushed/fsynced; local POSIX locks do not provide cross-Colab Google Drive locking. An incomplete final JSON fragment is warned about and ignored on read; append preserves its bytes and adds a hash-checked recovery marker. Arbitrary interior corruption is an error. Do not manually edit or truncate an original journal.

Identity/hash preparation failures (for example a missing checkpoint) occur before an experiment starts and raise directly. Data loading shared by a Grid also occurs before per-run attempts. The journal is a result/attempt log, not a full training checkpoint or crash-proof transactional database.

## Timing definitions (seconds)

| Field | Meaning |
|---|---|
| dense_scoring_time | Dense train-set loss/probability/feature pass, once per Damage attempt |
| pruned_scoring_time | Pruned/current train-set scoring at every non-Random/non-Full selection |
| scoring_time | Sum of dense and pruned scoring time |
| selection_time | Selection algorithm and its weight normalization; excludes scoring |
| train_time | Training loader/augmentation, loss, backward, optimizer and schedule |
| evaluation_time | Initial dense/pruned test evaluations plus all epoch test evaluations |
| total_wall_time | Run entry through final evaluation/metadata; excludes the final journal append |
| steps | Historical-compatible attempted training batches, including AMP-skipped updates |
| sgd_steps | Successful optimizer.step calls, counted with a post-step hook |
| skipped_optimizer_steps | Attempted batches minus successful optimizer steps |
| samples_processed | Total sample presentations to the training loop |
| unique_samples_seen | Exact union of training indices (not scoring/evaluation accesses) |

CUDA/MPS phase boundaries synchronize the device, and timers use perf_counter. Phase sums exclude setup, provenance/hash I/O, pruning, diagnostics and journal overhead; these are included in wall time within its stated boundary. Standalone runs include their own data loading; Grid per-run records explicitly mark data_preloaded because Grid setup is shared. Compare like scopes, GPUs and software environments. None of these timings are sparse-kernel FLOP measurements or dense pretraining cost.

Dense scoring has **no disk or cross-run cache**: computed once per Damage attempt and reused in memory across its re-selections. Its cost is always charged to that attempt. Random does no model scoring; its train_acc_before_select diagnostic is consequently omitted. This removes deterministic eval-only forward passes, not random draws. Extra instrumentation (sync, counters, hashing, logging) adds overhead and should not be compared to historical timing without qualification.

The loss, SGD defaults, masks, augmentation, selection definitions, schedule and final/best accuracy calculation remain unchanged. CPU regression tests compare all six methods to the archived trainer, including Random re-selection. GPU AMP/cuDNN numerical reproducibility is not asserted; validate on Colab. Optimizer hooks count updates without consuming torch RNG or introducing per-batch device synchronization.

## Aggregation and output safety

`src.plot.aggregate_records` groups all scientific settings except seed, plus checkpoint, source/protocol, timing schema, hardware and execution profile. Figures also separate epochs, weights, alpha and re-selection. Full-data references are drawn only within the same figure configuration. Familiar CSV columns remain at the front, with audit fields appended; missing legacy values remain unknown/NaN. All accuracy calculations use source values, without updating existing reports.

`src.plot` writes to a new directory and refuses to replace outputs unless explicitly --overwrite. `scripts/plot_reselect.py` is a frozen legacy-schema helper, import-safe, writes to a new default directory and refuses overwrites; use src.plot for new journals. CSV and figure aggregation from arbitrary provenance-free historical exports cannot recover missing experimental settings.

## Validation and remaining work

Run `python -m unittest discover -s tests -v` and `python scripts/build_notebook.py --check`. Tests use temporary artifacts and synthetic CPU tensors, not formal experiments. CUDA timing/AMP, actual CIFAR data, the shared pretrained checkpoint and Drive reconnect behavior require a Colab smoke test.

Deferred findings: CRAIG zero-cluster clamping/ties, mask-aware gradient proxy, BatchNorm recalibration, weight clipping/uniform-weight ablations, alpha sensitivity, cumulative coverage interpretation and more pretraining seeds. These are recorded but deliberately not changed in phase 1. Missing Drive artifacts must be requested from teammates; do not infer missing historical accuracy, environment or checkpoint hashes.

Phase-1 local validation used Python 3.12, PyTorch 2.4.0 and torchvision 0.19.0 on CPU (CUDA/MPS unavailable): all 25 unittest tests passed. The regression comparison covers all six training paths (full plus the five selectors), checks every final state_dict tensor and torch RNG state against the archived trainer, and separately checks Random re-selection. Active Notebook IPython-transformed cells compile, generated sources match, and all existing results files plus the archived Notebook were checked byte-for-byte against the starting main commit. This is synthetic correctness validation, not a new CIFAR accuracy result or a Colab/GPU performance measurement.
