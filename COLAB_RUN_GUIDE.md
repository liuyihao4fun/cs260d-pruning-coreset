# Colab：第一阶段实验审计版本

使用 `fix/experiment-audit` 分支和队友已有的 dense checkpoint。本指南不会训练 dense model，也不会启动完整历史网格。历史 CSV/图片及原 Notebook（`notebooks/colab_run_legacy.ipynb`）保持不变。

## 1. 选择 GPU、Clone 分支、安装依赖

Colab 菜单选择 **Runtime → Change runtime type → T4 GPU**。在一个新的 runtime 中执行：

```python
import pathlib, subprocess, sys
REPO = pathlib.Path('/content/cs260d-pruning-coreset')
BRANCH = 'fix/experiment-audit'
if not REPO.exists():
    subprocess.run(['git', 'clone', '--branch', BRANCH, '--single-branch',
                    'https://github.com/liuyihao4fun/cs260d-pruning-coreset.git', str(REPO)], check=True)
subprocess.run(['git', '-C', str(REPO), 'status', '--short', '--branch'], check=True)
COMMIT = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
print('实际 Git commit:', COMMIT)  # 保存这个值，用于断线恢复
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(REPO / 'requirements.txt')], check=True)
%cd /content/cs260d-pruning-coreset
```

如果仓库是 private，需要先通过自己有权限的 GitHub 认证方式 Clone；不要把访问 token 写入 Notebook、命令 URL 或提交到仓库。

已有 checkout 不会自动 pull 或覆盖。更新代码应在没有实验运行时操作，并为新版本选择新的结果文件。为了复现一次已开始的网格，应使用该次记录的 commit，而不是自动切换到分支最新提交。

## 2. 挂载自己的 Drive，指定已有 checkpoint 和独立结果文件

```python
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
from src.audit import file_sha256
import torch

WORK = Path('/content/drive/MyDrive/cs260d')  # 改为自己可访问的目录
CKPT = str(WORK / 'resnet18_cifar10.pt')
DATA = str(WORK / 'data')
NAME = 'teammate_name_t4'                    # 改成自己的名字/运行环境
RESULTS = str(WORK / f'results_audit_{NAME}_phase1.jsonl')
WORK.mkdir(parents=True, exist_ok=True)
assert Path(CKPT).is_file(), f'请向队友索取已有 checkpoint：{CKPT}'
assert torch.cuda.is_available(), '先启用 GPU runtime'
print('GPU:', torch.cuda.get_device_name())
print('Checkpoint SHA256:', file_sha256(CKPT))
print('Journal:', RESULTS, '(已存在则恢复，绝不清空)')
```

每位队友、每个并行 runtime 使用不同的 JSONL。**不要让两个 Colab 同时写同一个 Drive 文件。** 新文件名应与历史 `results_<name>.jsonl`、`results_diag.jsonl` 区分。

## 3. 先做 CPU 测试和 Notebook 一致性检查

```python
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)
subprocess.run([sys.executable, 'scripts/build_notebook.py', '--check'], check=True)
```

测试使用临时 synthetic tensors/checkpoint，不下载 CIFAR，不训练正式 dense model，不写历史结果。

## 4. 可选：极小 GPU smoke test

```python
!python -m src.finetune --ckpt "{CKPT}" --data_root "{DATA}" \
    --sparsity 0.95 --method random --frac 0.1 --epochs 1 --limit_train 2000 \
    --results "{RESULTS}" --tag smoke
```

这是 debug 协议，不能将其 accuracy 与完整 CIFAR 实验比较。`limit_train`、epochs 和 tag 均进入 identity。首次执行会下载 CIFAR 到 DATA（如果没有缓存）。

## 5. 单组正式实验

```python
!python -m src.finetune --ckpt "{CKPT}" --data_root "{DATA}" \
    --sparsity 0.95 --method random --frac 0.2 --epochs 30 --seed 0 \
    --results "{RESULTS}" --tag audit_phase1
```

再次执行相同命令会跳过已经完整完成的 v2 run。失败或中断的 run 会从 dense checkpoint 重新开始；不支持从某个 fine-tuning epoch 精确续训。

## 6. 小 Grid：先 Dry Run，再执行

下面是 **3 runs**，不是历史的 343 runs。第一条只显示配置和 identity，不加载/下载 CIFAR、不开始训练，但需要读取 checkpoint 来计算 hash。

```python
!python -m src.run_grid --ckpt "{CKPT}" --data_root "{DATA}" \
    --sparsities 0.95 --methods random craig damage_fl --fracs 0.2 --seeds 0 \
    --epochs 30 --results "{RESULTS}" --tag audit_phase1 --dry_run
```

确认后执行同一命令，去掉 `--dry_run`：

```python
!python -m src.run_grid --ckpt "{CKPT}" --data_root "{DATA}" \
    --sparsities 0.95 --methods random craig damage_fl --fracs 0.2 --seeds 0 \
    --epochs 30 --results "{RESULTS}" --tag audit_phase1
```

single run 与 Grid 中的 Random 配置相同，所以已完成时会跳过。需要 3 seeds 时明确改成 `--seeds 0 1 2`，变成 9 runs。

周期重选加 `--reselect_every 5`；Uniform Weights 加 `--no_weights`。它们有不同 identity，汇总时也分开；这只是命令说明，本阶段不自动启动这些消融。

## 7. 检查结果、成本和配置

```python
from src.audit import load_jsonl, completed_keys
records = load_jsonl(RESULTS)
completed = [r for r in records if r.get('status') == 'completed']
print('已完成 identities:', len(completed_keys(records)))
if completed:
    r = completed[-1]
    print(r['key'], r['status'], r['final_acc'])
    print(r['config'])
    print(r['metadata'])
    for k in ['scoring_time', 'dense_scoring_time', 'pruned_scoring_time',
              'selection_time', 'train_time', 'evaluation_time', 'total_wall_time',
              'steps', 'sgd_steps', 'unique_samples_seen', 'samples_processed']:
        print(k, r[k])
```

`selection_time` 在 v2 只包含选样算法；旧版本还包含 pruned scoring，两者不能直接比较。`total_wall_time` 包含 evaluation、准备及审计开销，截止最终记录写入之前；Grid 共享数据加载不计入单 run。不同 GPU、软件版本或计时范围的 wall time 不要直接作为公平对照。

画图使用新的输出目录：

```python
from datetime import datetime
OUT = str(WORK / f'audit_figures_{NAME}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
!python -m src.plot --results "{RESULTS}" --out "{OUT}"
print('查看 summary.csv，以及 figure_manifest.json 指向的各配置图片：', OUT)
```

不同 epochs、重选、weights、alpha、训练参数及硬件分别输出。默认拒绝覆盖已有图表。不要使用旧的全局 wildcard 无意混入所有队友/历史文件。

## 8. Colab 断线后恢复

1. 重启 runtime，重新挂载 Drive。
2. Clone 同一分支，并 checkout 之前保存的 **完整 commit hash**。不要在运行期间更新源码。
3. 重新安装依赖，恢复相同 CKPT、DATA、NAME、RESULTS 和所有实验参数。
4. 先 Dry Run，再执行同一 Grid。只跳过完整完成的 v2 runs；剩余任务重跑。

```python
# 将这里改为第 1 步保存的真实 hash（不要使用占位文本）
RESUME_COMMIT = 'PASTE_SAVED_FULL_COMMIT_HASH'
subprocess.run(['git', '-C', str(REPO), 'checkout', '--detach', RESUME_COMMIT], check=True)
print(subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True))
```

不要为了恢复删除 started/failed 记录，也不要编辑 JSONL。若断线留下最后一行不完整，读取时警告并忽略；再次追加会保留原始字节并添加 recovery marker。中间损坏且没有合法恢复标记时会报错，应保留原件并人工检查。

旧 JSONL 可用于读取和汇总，但因为缺少 v2 provenance，不会阻止新 run 执行。切勿为了节省重跑，把旧 key 手工替换成 v2 key。

## 9. 使用生成的 Notebook

`notebooks/colab_run.ipynb` 会 Clone 分支、验证源码指纹，并写入一致的 embedded src。Notebook 必须与 Clone 的版本一致；不一致时会停止。已有 dense checkpoint 不存在时会停止，不会重新训练。历史大网格均需显式设置 `RUN_LARGE_GRIDS = True` 才运行，第一阶段请保持关闭，按本指南选择小实验。

CLI 工作流也支持 detached commit 恢复，不依赖 embedded Notebook。旧 Notebook 仅作为历史档案，不用于运行新审计实验。

## 10. 队友需要提供的文件

- 共享的原始 `resnet18_cifar10.pt` 和 SHA256；如有多个版本，提供各自路径/hash及使用范围。
- 原始 `results_<name>.jsonl`（主实验、15/30 epochs、重选、3/6 epochs full baseline）。
- 原始 `results_diag.jsonl`，以及未并入上述文件的 smoke/失败运行日志。
- dense 训练日志、当时训练配置/seed；历史 notebook 输出（若保存）、GPU 和 PyTorch/CUDA 信息。
- 新版本的 `results_audit_*.jsonl`，包括 started/failed/interrupted/completed 记录，不要只导出 accuracy。

这些文件未包含在 Git 仓库内；维护者无法据此假设它们仍存在。缺失字段不要事后编造；明确标记 unknown。
