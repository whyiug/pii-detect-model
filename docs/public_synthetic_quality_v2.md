# Open-24 社区质量结果：生成与严格重放

`scripts/produce_public_synthetic_service_quality_v2.py` 将冻结的 24 类 gold、候选预测、
同轨 comparator 预测和性能清单转换成**只含聚合统计**的质量回执。它同时提供 strict replay：
重放时必须重新提供完全相同的输入，程序会校验每个字节哈希、重新逐文档配对、重新运行 bootstrap，
最后要求整份回执逐字段相等。

这条链路不读取模型权重、不查询或使用 GPU、不联网，也不读取 PII Bench。它只支持
`public_open_source + synthetic` 的公开测试已见、非生产社区评测；不能证明真实世界泛化，不能激活
“首个”“全球 SOTA”或“生产可用”声明。

> 证据边界：producer 会严格验证 gold dataset manifest、materialization/supersession receipt 和
> pre-result freeze receipt 的完整哈希链；candidate 还必须提供由冻结 v2 prediction-provenance
> 实现生成的 manifest，comparator 必须提供仓库/版本/许可证、生成回执、实现、运行配置和环境锁的
> content-addressed manifest。质量 producer 对这些清单做 metadata-only 验证并绑定文件哈希和自哈希，
> 但它本身不运行模型或 comparator，不能单独证明 prediction 确由所声明系统生成，也不能仅凭
> performance manifest 证明计时过程没有手工填写。因此 fixture、手工 prediction 或手工性能值不得
> 作为发布结果；正式发布门禁仍须完成 candidate full replay、comparator generation replay 和性能
> harness replay。回执明确记录 `prediction_origin_replay_verified_by_this_producer=false`、两个 replay
> required 标记、`performance_measurement_replay_verified_by_this_producer=false` 及
> `claim_activation_allowed=false`，不能把 metadata-only 通过冒充成来源或性能 harness 已重放。

## 为什么使用新的 comparator registry

历史 `comparator_registry_v1.yaml` 冻结的是 PII Bench closed-8，且没有 PII-free 文档。把其中的
聚合成绩当成 24 类 comparator 会混淆 gold、标签集合和 FPR 分母。因此 v2 使用独立的严格 schema：

- `configs/evaluation/public_synthetic_quality_comparator_registry_v2.schema.json`
- `scripts/build_public_synthetic_quality_registry_v2.py`

registry builder 只接受同一份 Open-24 gold 上的逐文档预测，并冻结 candidate/comparator prediction、
两侧 prediction provenance、性能清单、最终模型清单和服务配置清单的精确文件哈希与清单自哈希。
旧 registry 不会被修改，也不会被冒充为 24 类证据。
其中 `model_raw.candidate_id` 必须精确等于最终模型清单的 `model_id`，
`full_system.candidate_id` 必须精确等于服务配置清单的 `service_id`；不能用未绑定别名替代。

## 输入契约

Gold 直接接受项目 materializer 写出的 canonical `DocumentRecord` JSONL（`doc_id + text + entities`
以及 provenance/policy 字段），也兼容已经脱敏的 `doc_id + text_length + spans` 评测形式；不要求生成
另一个未绑定的转换文件。加载 canonical 行时，producer 会用原文临时校验 entity 文本切片和字符偏移，
随即只保留 `doc_id + text_length + spans`，不会把原文或实体值写入回执。span 是左闭右开
`[start, end)`，label 必须来自固定顺序的 24 类。整份 gold 必须让每一类至少有一个 gold span，
并至少有一个明确的 PII-free 文档；在本公开/合成协议中，无 entity/span 的行就是冻结的
PII-free partition。

Gold 不能裸文件进入评测。CLI 还必须同时提供 `dataset_manifest.json`、数据目录内的
materialization/supersession receipt、以及 manifest 指向的 pre-result freeze receipt，并固定
`--gold-split internal_evaluation`。producer/builder 会校验：manifest 与 receipt 自哈希、三者文件哈希、
dataset ID/version/config/freeze chronology、选中 split 的文件名/bytes/records/SHA-256、24 类与 PII-free
覆盖、零碰撞审计、`evaluation_only`/禁止训练和调参策略、以及 one-shot final-evaluation 边界。
这些约束从所提供 manifest 的闭合字段和哈希关系推导，不把某个测试 fixture 或外部绝对路径写进代码。

Candidate 与 comparator prediction JSONL 每行只能有 `doc_id` 和 `spans`，并且必须各自恰好覆盖
gold 的全部 doc ID，不允许少文档、额外文档或重复文档。`model_raw` 和 `full_system` 不能交叉比较。
每份 prediction 还必须有独立 provenance manifest。Candidate 使用
`configs/evaluation/release_eval_v2_prediction_provenance_v1.schema.json`；质量链路调用其 standalone
validator，逐项对齐 gold、最终训练清单、模型 artifact、calibration 和服务实现。Comparator 使用
`configs/evaluation/open24_comparator_prediction_provenance_v1.schema.json`，固定公开来源及六类生成环境
binding。两类清单都只含聚合元数据，不含路径、文本、实体值或 doc ID。

最终模型 binding manifest 是闭合 JSON：

```json
{
  "schema_version": "pii-zh.community-final-model-binding.v2",
  "model_id": "aiguard24-full-seed42",
  "artifact_class": "24_label_zh_hans_token_classifier",
  "label_count": 24,
  "ordered_labels": ["固定顺序的 24 个标签"],
  "taxonomy_version": "1.0.0",
  "attention_mode": "full",
  "training_manifest_file_sha256": "训练清单文件的 SHA-256",
  "training_manifest_sha256": "训练清单 canonical 自哈希",
  "model_identity_sha256": "完整模型身份的 SHA-256",
  "artifact_sha256": "模型发布 artifact 的 SHA-256",
  "manifest_sha256": "去掉本字段后 canonical JSON 的 SHA-256"
}
```

服务配置 binding manifest 必须声明 `canonical_track=full_system`，并通过 model ID、模型清单的
**文件 SHA-256**和 `model_identity_sha256` 绑定同一个最终模型。它还包含 `service_id`、`profile_id`、
`calibration_bundle_file_sha256`、`implementation_sha256`、实际配置 artifact 的
`configuration_sha256` 和同样计算的 `manifest_sha256`。

每个 track 的 performance manifest 必须绑定该 track 的 gold、candidate prediction、模型清单和
服务清单文件哈希：

```json
{
  "schema_version": "pii-zh.community-quality-performance-binding.v2",
  "track": "full_system",
  "system_id": "community-cascade-v2",
  "gold_sha256": "...",
  "predictions_sha256": "...",
  "final_model_manifest_sha256": "...",
  "service_configuration_manifest_sha256": "...",
  "measurement": {
    "sample_count": 10000,
    "latency_p50_ms": 10.0,
    "latency_p95_ms": 20.0,
    "latency_p99_ms": 25.0,
    "throughput_documents_per_second": 50.0,
    "throughput_characters_per_second": 10000.0,
    "peak_cpu_rss_mb": 1024.0,
    "peak_gpu_memory_mb": 4096.0
  },
  "manifest_sha256": "去掉本字段后 canonical JSON 的 SHA-256"
}
```

性能字段不会从 span 文件“猜测”；producer 验证其哈希和 track 绑定后原样纳入回执，使性能数据的
任何修改都令 strict replay 失败。`measurement.sample_count` 还必须等于 gold 的完整文档数，
不接受只测子集却作为整套 Open-24 性能结果的清单。

正式性能清单必须由
[`benchmark_release_eval_v2_candidate.py`](../scripts/benchmark_release_eval_v2_candidate.py)
真实复跑候选并验证 prediction bytes 后生成；质量回执生成后还必须再次执行其 `replay` 子命令，得到
community release v2 可消费的 `performance_harness_replay`。完整参数、先验 internal-unlock strict replay
和资源测量口径见
[`release_eval_v2_performance_harness.md`](release_eval_v2_performance_harness.md)。

## 冻结 registry

```bash
PYTHONPATH=src python scripts/build_public_synthetic_quality_registry_v2.py \
  --registry-id open24-community-release-1 \
  --gold /path/internal_evaluation.jsonl \
  --gold-dataset-manifest /path/dataset_manifest.json \
  --gold-materialization-receipt /path/supersession_receipt.json \
  --gold-freeze-receipt /repo/configs/evaluation/pre-result-freeze.json \
  --gold-split internal_evaluation \
  --final-model-manifest /path/final-model-binding.json \
  --service-configuration /path/service-binding.json \
  --track model_raw MODEL_ID MODEL_COMPARATOR_ID \
    /path/model.predictions.jsonl /path/model.prediction-provenance.json \
    /path/model-comparator.predictions.jsonl \
    /path/model-comparator.prediction-provenance.json \
    /path/model.performance.json \
  --track full_system SERVICE_ID SERVICE_COMPARATOR_ID \
    /path/service.predictions.jsonl /path/service.prediction-provenance.json \
    /path/service-comparator.predictions.jsonl \
    /path/service-comparator.prediction-provenance.json \
    /path/service.performance.json \
  --output /path/open24-comparator-registry.json
```

输出采用只读、no-clobber 发布。registry 自身不含成绩，也不激活 claim。

## 生成真实结果

```bash
PYTHONPATH=src python scripts/produce_public_synthetic_service_quality_v2.py produce \
  --gold /path/internal_evaluation.jsonl \
  --gold-dataset-manifest /path/dataset_manifest.json \
  --gold-materialization-receipt /path/supersession_receipt.json \
  --gold-freeze-receipt /repo/configs/evaluation/pre-result-freeze.json \
  --gold-split internal_evaluation \
  --final-model-manifest /path/final-model-binding.json \
  --service-configuration /path/service-binding.json \
  --comparator-registry /path/open24-comparator-registry.json \
  --track model_raw /path/model.predictions.jsonl \
    /path/model.prediction-provenance.json \
    /path/model-comparator.predictions.jsonl \
    /path/model-comparator.prediction-provenance.json \
    /path/model.performance.json \
  --track full_system /path/service.predictions.jsonl \
    /path/service.prediction-provenance.json \
    /path/service-comparator.predictions.jsonl \
    /path/service-comparator.prediction-provenance.json \
    /path/service.performance.json \
  --samples 10000 --seed 42 \
  --output /path/open24-quality-result.json
```

程序重算以下内容：

- 24 类 exact `doc_id + label + start + end` strict span micro/macro 与逐类 P/R/F1；
- PII-free document FPR 与每千 PII-free 字符 FP span；
- label-agnostic PII 区间 union 的漏检字符率和非 PII 字符误标率；
- candidate 对同轨 comparator 的至少 10,000 次 paired document percentile bootstrap；
- v1 社区合同中原有的质量、FPR、字符错误、资源和 bootstrap 阈值，数值不作修改。

`full_system` 仍是 release-blocking track，`model_raw` 仍是非阻塞 track。回执中的
`benchmark_leading_on_named_open24_suite` 只是一项同协议描述性结果；`claim_activation_allowed`
始终为 `false`，后续发布合同必须单独决定允许的文案范围。

## 严格重放

```bash
PYTHONPATH=src python scripts/produce_public_synthetic_service_quality_v2.py replay \
  --gold /path/internal_evaluation.jsonl \
  --gold-dataset-manifest /path/dataset_manifest.json \
  --gold-materialization-receipt /path/supersession_receipt.json \
  --gold-freeze-receipt /repo/configs/evaluation/pre-result-freeze.json \
  --gold-split internal_evaluation \
  --final-model-manifest /path/final-model-binding.json \
  --service-configuration /path/service-binding.json \
  --comparator-registry /path/open24-comparator-registry.json \
  --track model_raw /path/model.predictions.jsonl \
    /path/model.prediction-provenance.json \
    /path/model-comparator.predictions.jsonl \
    /path/model-comparator.prediction-provenance.json \
    /path/model.performance.json \
  --track full_system /path/service.predictions.jsonl \
    /path/service.prediction-provenance.json \
    /path/service-comparator.predictions.jsonl \
    /path/service-comparator.prediction-provenance.json \
    /path/service.performance.json \
  --receipt /path/open24-quality-result.json
```

Gold、任一 candidate/comparator prediction、两侧 provenance 清单、性能清单、model/service binding、
registry hash、track 或实现代码 hash 只要变化，重放就会 fail closed。回执 schema 位于
`configs/evaluation/public_synthetic_service_quality_result_v2.schema.json`，禁止额外字段；回执不含
路径、原始文本、doc ID、实体值或逐文档统计。

## Open24 comparator 的实际生成与 execution replay

同一份 release-eval v2 internal gold 上的两个正式 comparator 不再接受手工拼装的六个 metadata
binding。生成器、固定系统语义、stage pre-open gate、one-descriptor input snapshot、模型文件校验、
measurement harness 和 full execution replay 保留在内部历史规范
`docs/release_eval_v2_open24_comparators.md`。正式门禁应分别要求：

- `model_raw`：pinned original AIguard Open24 prediction bundle；
- `full_system`：native Presidio + pinned CLUENER + `cn_common_v6` bundle；
- 两者的 provenance strict replay；
- 两者重新推理后的 prediction byte equality receipt；
- 两者重新采集 measurement 的 harness replay。

只有 comparator manifest 的 schema/self-hash 通过仍然是 metadata-only；缺少任一 execution replay
receipt 时必须保持 BLOCK。
