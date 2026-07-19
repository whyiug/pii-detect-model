# release-eval-v2 候选性能测量与严格重放

[`scripts/benchmark_release_eval_v2_candidate.py`](../scripts/benchmark_release_eval_v2_candidate.py)
为冻结的 Open-24 候选生成
`pii-zh.community-quality-performance-binding.v2`，并在质量结果生成后产生社区发布 v2 所要求的
`performance_harness_replay` receipt。它支持：

- `model_raw`：选中模型、threshold=0、未校准；
- `model_calibrated`：同一模型和 calibration 的 model-only 诊断轨；
- `full_system`：`community-model-cascade-v1` 的模型、v6 规则、路由和 validators 完整级联轨。

只有 `model_raw` 和 `full_system` 是社区质量/发布合同的 canonical tracks；
`model_calibrated` 可以生成性能 manifest 和做本地重测，但不能冒充发布合同中的第三条轨。

## 正式输入的先验门禁

正式 `internal_evaluation` 不是只检查一个自签 JSON。每次 `measure` 和 `replay` 都会在首次打开、哈希或
解码 internal JSONL **之前**调用 `replay_internal_unlock`，从以下原始证据重建完整链路并要求逐字段相等：

1. 冻结的三 seed selector protocol、development manifest、v1 release manifest 和三个 candidate；
2. selector receipt 和 v1→v2 amendment；
3. v2 dataset manifest、materialization/supersession receipt 和 freeze receipt；
4. calibration authorization、calibration gold、raw fit prediction、generation receipt、prediction manifest、
   calibration bundle 和 diagnostics；
5. final model binding、service configuration binding 和 internal unlock。

严格重放通过后，`guard_internal_preopen` 再做一次 split 名称、文件大小、模式和 binding 的 pre-open 检查。
此后输入通过 `O_NOFOLLOW` 文件描述符读取，并比较读前/读后 inode、size、mtime、ctime，内存中的 bytes
就是本次执行的稳定 snapshot。已冻结 candidate prediction 也使用相同 snapshot 机制。模型 artifact、
calibration 和仓库实现会在运行前绑定；模型与实现身份在运行后再次核验。

性能 replay 只接受 `pii-zh.community-service-source-identity.v4`。harness 会独立重建当前完整
`src/pii_zh` source inventory，核验 v4 自哈希与 inventory hash/count，并要求它相对不可变 v3 前驱的
`successor_delta` 必须精确包含以下 5 项 changed：
`pii_zh.evaluation.release_eval_v2_performance.py`，以及
`pii_zh.data.synthetic.sota_v1.py`、`pii_zh.data.synthetic.sota_v2_resplit.py`、
`pii_zh.data.synthetic.sota_release_eval_v1.py`、
`pii_zh.data.synthetic.sota_release_eval_v2.py` 四项 packaging-portability amendment。后四项只把固定本机
绝对默认路径改为 `$PII_ZH_DATA_ROOT`（缺省为相对 `data/`）；added/removed、其他 changed 或显式 CLI
路径行为变化都会失败关闭。v4 由 community evidence producer 消费 mode `0444` 的 v3 前驱生成，不能
手工把旧 v2/v3 source manifest 传给 performance replay，也不能在源码继续漂移后复用旧 v4。
performance replay 的 `implementation_identity_sha256` 同时绑定 source schema version、完整 runtime
source map、inventory hash/count/complete；即使变化的 package 文件不在显式 component 列表中，也会改变
implementation identity 并使旧回执失效。

## 测量口径

测量参数固化在实现中，CLI 不接受性能数值，也不接受 batch/预热数值：

- document batch size：16（与冻结 candidate generation receipt 一致）；
- model micro batch size：8（与冻结 candidate generation receipt 一致）；
- warmup：前 2 个 batch；
- max tokens：512；stride fraction：0.25；
- 单进程、单卡（CUDA 时必须只有一个 `CUDA_VISIBLE_DEVICES` 条目）；
- p50/p95/p99：`perf_counter_ns` 测得的 batch detector 调用耗时按 batch 文档数归一化；
- docs/s、chars/s：同一次完整 corpus pass 的端到端 wall time，包含 prediction 规范化与序列化；
- peak RSS：`resource.getrusage(RUSAGE_SELF).ru_maxrss` 进程生命周期高水位；
- CUDA peak：reset 后的 `torch.cuda.max_memory_allocated`；CPU 运行固定为 0。

执行结果必须用同一 `write_prediction_jsonl` 规范序列化，并与冻结 candidate prediction **逐字节相等**。
不相等时不生成性能证据。性能 manifest 绑定 gold bytes、prediction bytes、final model binding 和 service
binding 的文件 SHA-256；输出不含路径、原文、实体值或 document ID，使用 no-clobber、原子硬链接发布和
`0444` 权限。

`replay` 会重新加载真实模型/服务并再次完整执行，而不是重算 metadata。prediction bytes 仍必须完全相等；
性能值因为共享服务器调度噪声不要求逐值相等，但两次测量都必须满足：百分位单调、docs/s 与 chars/s
来自同一 wall clock、sample count 覆盖全部文档、CPU/GPU resource class 一致，且固定实现/运行时下各项
正值不能相差超过 100 倍。成功输出严格符合
`configs/release/community_cascade_release_v2.replay-receipt.schema.json`，其 core 与 release v2 validator
独立推导的 `performance_harness_replay` 完全一致。

## 正式 measure

在公共 GPU 服务器上先只读执行 `nvidia-smi`，选空闲卡；不得 kill、暂停、reset 或抢占其他任务。以下
示例把只读查询后选出的空闲物理卡 `<FREE_GPU_ID>` 隔离成进程内的 `cuda:0`。`model_raw` 只需把 `--track` 改为 `model_raw`；正式两轨必须
分别执行、分别输出。所有输出路径必须尚不存在。

```bash
CUDA_VISIBLE_DEVICES=<FREE_GPU_ID> \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
PYTHONPATH=src python scripts/benchmark_release_eval_v2_candidate.py measure \
  --track full_system \
  --target-split internal_evaluation \
  --model /path/selected-model \
  --calibration /path/calibration-bundle.json \
  --input /path/internal_evaluation.jsonl \
  --expected-predictions /path/full-system.predictions.jsonl \
  --final-model-binding /path/final-model-binding.json \
  --service-configuration-binding /path/service-configuration-binding.json \
  --device cuda:0 \
  --protocol configs/train/aiguard24_sota_v3_resplit_selection.json \
  --development-manifest /path/development-dataset-manifest.json \
  --v1-release-manifest /path/release-eval-v1-dataset-manifest.json \
  --candidate /path/seed13-v3 \
  --candidate /path/seed42-v3 \
  --candidate /path/seed97-v3 \
  --selection-receipt /path/aiguard24-v3-selection.json \
  --amendment /path/release-eval-v2-amendment.json \
  --dataset-manifest /path/release-eval-v2/dataset_manifest.json \
  --materialization-receipt /path/release-eval-v2/supersession_receipt.json \
  --freeze-receipt configs/evaluation/synthetic_sota_release_eval_v2_supersession_freeze.json \
  --calibration-gold /path/release-eval-v2/calibration.jsonl \
  --calibration-authorization /path/calibration-authorization.json \
  --calibration-diagnostics /path/calibration-diagnostics.json \
  --calibration-fit-predictions /path/calibration-model-raw.predictions.jsonl \
  --calibration-fit-generation-receipt /path/calibration-model-raw.generation.json \
  --calibration-fit-prediction-manifest /path/calibration-model-raw.provenance.json \
  --internal-unlock /path/internal-unlock.json \
  --output /path/full-system.performance.json
```

## 正式 replay

质量 producer 把上述 performance manifest 纳入 Open-24 quality receipt 后，再运行：

```bash
CUDA_VISIBLE_DEVICES=<FREE_GPU_ID> \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
PYTHONPATH=src python scripts/benchmark_release_eval_v2_candidate.py replay \
  --track full_system \
  --target-split internal_evaluation \
  --model /path/selected-model \
  --calibration /path/calibration-bundle.json \
  --input /path/internal_evaluation.jsonl \
  --expected-predictions /path/full-system.predictions.jsonl \
  --final-model-binding /path/final-model-binding.json \
  --service-configuration-binding /path/service-configuration-binding.json \
  --device cuda:0 \
  --protocol configs/train/aiguard24_sota_v3_resplit_selection.json \
  --development-manifest /path/development-dataset-manifest.json \
  --v1-release-manifest /path/release-eval-v1-dataset-manifest.json \
  --candidate /path/seed13-v3 \
  --candidate /path/seed42-v3 \
  --candidate /path/seed97-v3 \
  --selection-receipt /path/aiguard24-v3-selection.json \
  --amendment /path/release-eval-v2-amendment.json \
  --dataset-manifest /path/release-eval-v2/dataset_manifest.json \
  --materialization-receipt /path/release-eval-v2/supersession_receipt.json \
  --freeze-receipt configs/evaluation/synthetic_sota_release_eval_v2_supersession_freeze.json \
  --calibration-gold /path/release-eval-v2/calibration.jsonl \
  --calibration-authorization /path/calibration-authorization.json \
  --calibration-diagnostics /path/calibration-diagnostics.json \
  --calibration-fit-predictions /path/calibration-model-raw.predictions.jsonl \
  --calibration-fit-generation-receipt /path/calibration-model-raw.generation.json \
  --calibration-fit-prediction-manifest /path/calibration-model-raw.provenance.json \
  --internal-unlock /path/internal-unlock.json \
  --performance-manifest /path/full-system.performance.json \
  --quality-receipt /path/open24-quality-result.json \
  --service-source-manifest /path/community-service-source-identity.v4.json \
  --output /path/full-system.performance-replay.json
```

成功终端输出只有 aggregate hash、track 和状态。把其中的 `implementation_file_sha256` 与
`implementation_identity_sha256` 精确写入 community release v2 合同对应 track 的
`performance_harness_replay` binding；不能手工换成别的 harness/hash。

## Fixture 与边界

`--target-split fixture` 只用于单元/集成测试，拒绝所有 formal stage 参数；它仍要求真实 model、calibration、
final model binding、service binding 和冻结 prediction bytes。fixture manifest 不是正式性能证据，不能进入
quality registry。正式命令不会联网、不会打印任何行，也不会自动选 GPU；GPU 选择必须遵守共享服务器规则。
