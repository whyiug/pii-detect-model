# 中文 PII 级联统一评测

`scripts/evaluate_cascade.py` 把发布 `CascadePipeline` 与仓库现有 exact-span evaluator 接到一起，
用于消除“benchmark 复制一套 fusion、产品运行另一套逻辑”的分叉。

## 当前可执行范围

原入口仍支持 `rules-only`、`model-only` 和 `cascade` 三条发布运行轨。六组消融则由独立的安装后入口
`pii-zh-evaluate-ablation` 和版本化 packaged profiles 表达，具体 stage 矩阵、BYO NER binding 与
fail-closed 规则记录在内部历史规范 `docs/cascade-ablation-profiles.md`。六个 adapter 已有同一
`CascadePipeline` 执行路径，不代表六组 human hidden 结果已经产生：D1 threshold/context freeze、BYO
NER artifact、gate-selected project checkpoint、human hidden 物化和 custodian 执行仍是独立门禁。

六路入口要求显式 `--run-intent`。开发 smoke 固定 non-claim；正式运行还会在读取 hidden 内容前验证
adapter source closure/runtime lock。项目模型正式轨另逐字节交叉验证 candidate freeze、checkpoint
selection、release-gate 三份 receipt；NER 正式轨验证公开 model/revision/license admission receipt。六路
scope 固定为 24 类，不允许用 `--entity` 缩小分母。预测 document ID 会稳定哈希，predictions/report
权限为 `0400`；单路报告在 suite-level hidden/完整性/统计审核前始终 `claim_eligible=false`。

## 输入与运行

输入是项目 canonical JSONL：每行包含 `DocumentRecord` 的 `doc_id`、`text`、exact spans、来源与质量
元数据。仓库中的 [`../examples/data/synthetic_inputs.jsonl`](../examples/data/synthetic_inputs.jsonl)
只使用示例域名和 IANA 文档保留地址，可用于 smoke；它不是 benchmark，也不能支持效果声明。

```bash
python scripts/evaluate_cascade.py \
  --input examples/data/synthetic_inputs.jsonl \
  --predictions runs/cascade/rules.predictions.jsonl \
  --report runs/cascade/rules.report.json \
  --mode rules-only \
  --bootstrap-samples 1000 \
  --seed 42
```

本地模型轨：

```bash
python scripts/evaluate_cascade.py \
  --input /approved/evaluation.jsonl \
  --predictions runs/cascade/model.predictions.jsonl \
  --report runs/cascade/model.report.json \
  --mode model-only \
  --model-path /absolute/path/to/release-model \
  --device cpu
```

`--model-path` 必须是已有本地目录；脚本不接受 Hub ID 或 URL，不提供联网回退。共享 GPU 上如需模型
评测，仍必须在运行前重新查询 `nvidia-smi`，由调用方用 `CUDA_VISIBLE_DEVICES=<空闲卡>` 限制设备，
不得停止或抢占其他进程。

## 输出合同

- predictions 只含 `doc_id` 与 `(start, end, label, score)`，不含原文或实体值；
- report 复用 strict/relaxed/character/per-label/bootstrap 指标实现；
- report 绑定 mode、完整路由 config 与 SHA、级联实现 SHA、prediction SHA；
- 模型模式另绑定训练 manifest 文件 SHA 和可选 calibration SHA；
- 新输出以同目录临时文件生成、硬链接发布并设为只读；已有目标文件不会被覆盖；
- CLI 错误使用固定安全文案，不回显非法输入或实体值。

这些属性保证结果可追溯，但不会自动把 development 数据变成独立证据。报告是否可用于发布声明仍取决于
dataset revision、hidden 访问记录、comparator freeze、许可证和候选冻结时间。

## 必报机器指标定义

`scripts/evaluate_cascade.py` 通过独立 additive 模块在级联报告中新增顶层对象
`required_metrics`，schema 为 `pii-zh.required-metrics.v1`。共享 `evaluate_documents` 的返回 schema 和
实现保持不变；原有 `strict`、`relaxed`、
`character`、`boundary`、`pii_free`、`confidence_calibration`、`confusion` 和 `bootstrap` 的字段与
定义保持不变。

### T0/T1 minimum recall

- 基础值是现有 strict exact `(start, end, label)` 的逐标签 recall；
- label-to-tier 来自打包 taxonomy 的 output labels，并输出 taxonomy version、T0/T1 assignment 和
  assignment SHA-256；
- 与训练/校准门一致，若一个标签同时声明 T0/T1，则采用 **T0 precedence**，只进入 T0；
- 每个 tier 的值是该 tier 中在当前 gold 上 `support > 0` 的标签 recall 最小值，同时报告达到最小值的
  label 和 gold support；
- 当前 evaluator 没有独立的 frozen label-scope manifest，因此不能把 gold 中完全无支持的标签猜成
  recall 0。某 tier 没有 gold-supported label 时，该 tier 明确返回 `unavailable`；这不能用于宣称通过
  tier guardrail。

### PII-free false-positive spans / 1K characters

只选择 gold span 为空的文档：

```text
false_positive_spans_per_1000_characters
  = emitted_prediction_span_count / sum(gold.text_length) * 1000
```

PII-free 文档上的每个 emitted span 都计一个 FP span；重叠 span 不按字符合并。若没有 PII-free 文档、
任一 PII-free 文档缺 `text_length`，或总字符数为 0，则 ratio 为 `null` 并返回明确的 `unavailable`
reason。脚本不会使用最大 span end 猜测文档长度。

### Missed-sensitive / over-redaction character rate

这两个指标是面向最终遮盖效果的 label-agnostic 视图。每个文档先分别合并 gold span 和 prediction span
的字符区间，记为 `G` 和 `P`，跨文档累加：

```text
missed_sensitive_character_rate = |G - P| / |G|
over_redaction_character_rate   = |P - G| / |P|
```

因此预测边界正确但 label 错误时，strict/character label 指标仍计错，而 redaction 视图认为敏感字符已被
覆盖。若 `|G|` 或 `|P|` 为 0，对应 rate 返回 `unavailable`，不猜成 0。

### Fragmentation rate

一个 gold span 若与至少两个 **同 label** 的 emitted prediction span 有正字符交集，则该 gold span 计为
fragmented：

```text
fragmentation_rate = fragmented_gold_span_count / gold_span_count
```

不同 label 的重叠属于 confusion，不计 fragmentation。没有 gold span 时返回 `unavailable`。所有新增
输出都只含聚合计数、label、状态和哈希，不含原文、实体值或 document ID。

## 发布前六组消融

在人类 hidden 解封前，必须冻结并依次运行：

1. `rules_only`；
2. `Presidio + 中文 NER + cn_common`；
3. `project_model_raw`；
4. `simple_union`；
5. `validator_threshold_fusion`；
6. `final_cascade`。

六组必须使用相同文档、标签 scope、offset 合同和 metric implementation。除了 strict micro/macro F1，
还要报告 PII-free 文档 FPR、每千字符误报、逐标签最小召回、模型新增候选 precision、从 baseline FN
恢复的比例、paired bootstrap 差值、模型调用率和资源成本。

当前 PII Bench ZH 和 v6 合成验证都已经参与诊断，不能承担这次 confirmatory 结论。新的 headline 只能
来自尚未被训练方查看的 human hidden revision；否则结果必须标为公开回归或诊断。

CPU/rules-only 的最小资源 smoke 见
[`cascade-resource-benchmark.md`](cascade-resource-benchmark.md)。它直接调用同一发布 pipeline，并报告
冷/热口径、p50/p95、吞吐、模型调用率和进程 RSS，但结果仅为 `DIAGNOSTIC_ONLY`，不能替代冻结硬件、
checkpoint 和 workload 后的 C6 发布资源报告。
