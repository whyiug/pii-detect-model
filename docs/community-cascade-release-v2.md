# 社区模型与级联服务发布门禁 v2

`community_cascade_release_v2` 是 `community_cascade_release_v1` 的后继合同。它不修改、覆盖或
重新解释历史 v1。v2 的唯一用途，是在 Open-24 公开/合成评测完成后，对“中文 PII 模型 +
community-model-cascade-v1 集成服务”的本地候选做失败关闭验收。

当前仓库内的 [`community_cascade_release_v2.json`](../configs/release/community_cascade_release_v2.json)
故意保持 `blocked_pending_replays_and_artifacts`：正式 quality、replay、模型、服务及发布 artifact
binding 都是 `null`。这不是失败，也不能手工改成 PASS；它表示尚未生成正式发布回执。

## 为什么 quality v2 的 PASS 还不够

`public_synthetic_service_quality_result_v2` 会重算严格 span 指标、PII-free FPR、字符错误和 paired
bootstrap，但它明确只对输入清单做 metadata 验证。其
`candidate_prediction_full_replay_required`、`comparator_generation_replay_required` 和
`performance_harness_replay_required` 字段只是待办声明，不是执行证据。v2 发布门禁因此要求七份彼此
独立、路径无关、自哈希的 exact replay receipt：

1. `quality_strict_replay`：用冻结输入逐字段重建 quality v2 回执；
2. `model_raw:candidate_prediction_full_replay`；
3. `model_raw:comparator_generation_replay`；
4. `model_raw:performance_harness_replay`；
5. `full_system:candidate_prediction_full_replay`；
6. `full_system:comparator_generation_replay`；
7. `full_system:performance_harness_replay`。

两份 performance replay 必须由
[`benchmark_release_eval_v2_candidate.py`](../scripts/benchmark_release_eval_v2_candidate.py)
在完整 internal-unlock 上游 strict replay 通过后重新运行真实候选生成；不能用手写性能 manifest 或
metadata-only receipt 替代。运行手册见
[`release_eval_v2_performance_harness.md`](release_eval_v2_performance_harness.md)。

每份回执都必须符合
[`community_cascade_release_v2.replay-receipt.schema.json`](../configs/release/community_cascade_release_v2.replay-receipt.schema.json)，
状态只能是 `EXACT_REPLAY_PASS`，并同时绑定：quality 文件哈希及自哈希、目标 manifest 文件哈希及
自哈希、最终 model/service/calibration/source 身份、重放实现文件哈希与实现身份、完整输入集和输出集
哈希。validator 会从 quality 和目标 manifest 独立推导这些字段，再逐字段比较；不会把 metadata bool
转换成 PASS。

quality、candidate 和 comparator 的 gate wrapper 还必须绑定一份 native replay evidence；绑定值进入
wrapper 的 `input_set_sha256`，validator 会从 invocation-only 的 native evidence 路径重新计算。正式 producer
是
[`produce_community_cascade_release_v2_evidence.py`](../scripts/produce_community_cascade_release_v2_evidence.py)：

- quality native evidence 只在 `produce_public_synthetic_service_quality_v2.py replay` 对冻结输入逐字段重建成功
  后生成；
- candidate native evidence 会调用既有 frozen generator 在临时目录重新推理，要求 prediction bytes、
  generation receipt bytes 和重建 provenance 全部相等；现有 metadata-only provenance replay 不能单独签发；
- comparator wrapper 只消费通过 closed schema、自哈希、non-fixture 和 admitted bundle 全量交叉绑定的
  `open24_comparator_full_execution_replay`；不会把 stdout 中的 `PASS` 当证据。

native evidence 和 wrapper 均 path-free、自哈希、no-clobber、`0444`。wrapper 的 implementation identity 同时
绑定 producer 自身和实际 quality/generator/provenance/comparator implementation；修改其中任一文件后必须
重新执行 replay，不能只重签合同。

此外，正式 internal evaluation 在任何 producer/generator 打开 gold 之前，必须先通过上游
`release_eval_v2_internal_preopen_unlock`。该时序门由 release-eval-v2 stage gate 在入口处执行；本发布
门禁会强制消费并绑定其最终回执以及 model/service/calibration/diagnostics 身份，但不能事后补救“先打开
gold、后验证 unlock”的运行。它是七份 replay receipt 之外的第八项必需前置证据。

## 身份闭合

门禁会同时读取以下证据，不读取预测 JSONL 或 gold JSONL。最终制品源闭包会对实际模型包逐文件
哈希（因此会读取模型权重字节），但不加载模型推理、不查询/使用 GPU，也不联网：

- quality v2 聚合回执；
- `community-final-model-binding.v2`；
- `community-service-configuration-binding.v2`；
- calibration bundle；
- `community-service-source-identity.v4`；
- quality/candidate/comparator native replay evidence；
- 两轨 candidate/comparator prediction provenance manifest；
- 两轨 performance manifest；
- 七份 replay receipt；
- 发布 artifact 和本地 smoke/test 回执。

必须满足的等式包括：quality 的 model/service ID、模型训练清单 hash、模型 identity、模型 artifact、
服务配置 hash、calibration bundle hash、服务 implementation hash，都与 full-system candidate provenance
和最终 binding 一致；calibration 的 `model_version` 必须等于选中模型的 training-manifest 自哈希；source
manifest 必须逐项复现 full-system provenance 中的 `implementation_source_sha256`。v4 还独立绑定完整
`src/pii_zh` 可安装包源码 inventory；validator 会从当前树重新枚举并逐文件校验，因此
`fusion/deterministic.py`、`fusion/replacement.py`、`cascade/context.py`、`service/app.py` 以及 taxonomy
配置等传递依赖不能游离在服务身份之外。v4 不覆写已经发布为 `0444` 的 v3，而是绑定 v3 的文件 hash、
自哈希和 runtime inventory 身份，并记录闭合的 `successor_delta`。首次 v4 freeze 唯一允许的 delta 是
performance harness 与四个 synthetic materializer 的 packaging-portability amendment：后者只把固定
本机绝对默认值改为 `$PII_ZH_DATA_ROOT`（缺省为相对 `data/`），不改变显式 CLI path、模型、校准、
规则、路由或服务行为。新增、删除或修改其他 package source 都会失败关闭。任一不一致都会报错，不能
降级成 rules-only 或继续发布。历史 v2/v3 source identity 均保留不覆写。

## 合同与输出的路径边界

合同和最终 release receipt 只保存 logical ID 与 SHA-256，不保存本地路径。路径只通过本次 CLI 参数
传入。输出在写入前还会递归拒绝路径字段或路径字符串。

指定 `--output` 时，validator 使用临时文件、硬链接式 no-clobber 发布并把结果设为 `0444`；目标已
存在（包括 symlink）时直接失败。未指定 `--output` 只向 stdout 输出路径无关报告。validator 不联网、
不查询/使用 GPU、不读取 record-level 数据，也不执行 GitHub/Hugging Face 发布。它会为物理制品闭包
读取并哈希模型权重；回执中的 `model_weights_read` 会如实记录这一点。

## 类型化发布制品与验证回执

v2 不再把任意文件的 SHA-256 当作发布制品或 `PASS` 证明。以下十项必须由
[`produce_community_cascade_release_v2_artifacts.py`](../scripts/produce_community_cascade_release_v2_artifacts.py)
生成，并通过 closed JSON Schema、自哈希、producer/schema identity 和跨制品关系校验：

1. `model_card`：从正式 quality/model/service 回执机械渲染的发布前模型卡；同一 producer 调用必须以
   no-clobber、`0444` 成对输出 typed JSON 和字节完全一致的 Markdown companion，不能手工从 JSON 抽取；
2. `model_package_manifest`：实际 Hugging Face safetensors 模型目录的完整清单；
3. `technical_documentation_manifest`；
4. `wheel_manifest`；
5. `wheelhouse_manifest`：离线依赖 wheelhouse 的逐 wheel、逐分发版本闭包；
6. `container_manifest`；
7. CycloneDX 1.6 `sbom`；
8. `license_report`；
9. `public_artifact_scan`；
10. `dependency_scan`。

模型卡固定说明候选仍为 `unpublished_local_candidate`，Python 发布候选版本固定为
`pii-zh-qwen==0.2.0rc1`。许可证报告只能登记事实清单和
`COMPLETE_HUMAN_APPROVAL_PENDING`，不能自行伪造法律批准；2026-07-20 的维护者批准由独立外部回执
提供并绑定最终文件。依赖扫描必须运行固定的 OSV Scanner v2 命令并得到零 finding。该网络查询只
发生在 dependency artifact producer；最终 gate 验证冻结结果、scanner 身份和 SBOM 关系，不重新
运行会随时间变化的 OSV 查询。container 必须同时绑定 wheel、
wheelhouse 与模型包，container smoke 还会用 `importlib.metadata` 核对镜像内分发版本。public scan
深扫项目模型包、项目 wheel、文档和类型化 JSON；第三方 dependency wheel 只做文件哈希/包版本闭包，
其许可证和漏洞风险分别由 license/SBOM/OSV 管理。

五项验证由
[`produce_community_cascade_release_v2_verifications.py`](../scripts/produce_community_cascade_release_v2_verifications.py)
执行固定 harness：当前社区 RC 回归套件、clean-wheel smoke、无网络 container smoke、CPU/offline 模型
smoke 和 CPU/offline 服务 smoke。第一项不是“全仓测试全绿”声明：它使用源码中闭合的显式测试文件
allowlist，覆盖核心 service/cascade/inference/fusion/rules/presidio/calibration、release-eval-v2、community-v2
证据/制品/验证/合同、build/public-scan/templates 和打包可移植性；历史冻结门仍作为历史证据保留，但不混入
当前 successor 的 RC gate。该 receipt 额外绑定 allowlist、pytest 支持文件，以及被测 release-critical
producer/module/schema 的逐文件内容清单 hash 和文件数；harness 在执行前后重算完整清单，任一测试、配置
或被测实现运行中漂移都会失败关闭，签发前还会再次核对同一身份。所有 v2 receipt 均绑定固定 argv 的
语义 hash、harness/producer/schema identity、返回码、
脱敏 stdout/stderr/log hash 和精确 subject；caller 不能传入 `PASS` 或自报 evidence hash。clean wheel 与
service smoke 固定安装 release wheel 的 `[cascade,service]` extras，model smoke 固定安装 `[inference]`；
三者均使用与 `wheelhouse_manifest` 完全一致的物理目录和 `--no-index`。

完成合同必须由
[`build_community_cascade_release_v2_contract.py`](../scripts/build_community_cascade_release_v2_contract.py)
从路径自动组装。builder 不接受 caller 提供的 hash、状态或 publication receipt；它会派生全部 binding，
在临时合同上调用同一正式 gate，并且只在状态精确为 `READY_FOR_USER_AUTHORIZATION` 时以 no-clobber、
`0444` 发布合同。该状态仍只是本地候选完成，绝不等于 GitHub/Hugging Face 已发布或已获授权。

## 正式命令骨架

以下只是一份 builder 参数骨架；`NAME=PATH` 中的 NAME 必须与闭合 inventory 精确一致。所有制品和
验证回执应先由上述固定 producer 生成，不能手工填 hash 或 PASS，也不要修改历史 v1/v2 证据。

模型卡必须先用同一固定命令同时物化 JSON 证据与模型包 `README.md`；若任一目标已存在或第二个输出
失败，命令失败关闭且不会留下半套新输出：

```bash
PYTHONPATH=src python scripts/produce_community_cascade_release_v2_artifacts.py model-card \
  --quality-receipt /path/open24-quality-result-v2.json \
  --final-model-manifest /path/final-model-binding.json \
  --service-configuration-manifest /path/service-configuration-binding.json \
  --training-manifest /path/aiguard24-full-seed97/training_manifest.json \
  --pii-bench-report /path/seed97-pii-bench-zh-posthoc.json \
  --template-asset src/pii_zh/data/synthetic/assets/curated_templates_v1.json \
  --output /path/model-card.evidence.json \
  --markdown-output /path/model-card.README.md
```

seed-97 社区模型包使用专用的最小 assembler；它不接受 legacy `--evidence-dir`、旧 Model Card bundle、
seed-42 evaluation report 或 `--force`。`id2label.json` 从 checkpoint config 机械规范化并与仓库 core-24
taxonomy 核对，`training_manifest.json` 必须来自同一 checkpoint 且与实际权重/配置完全绑定；校准和质量
证据由外部 typed artifacts/社区合同绑定，不伪装成模型包内训练证据：

```bash
PYTHONPATH=src python scripts/build_release.py \
  --checkpoint-dir /path/aiguard24-full-seed97 \
  --model-card /path/model-card.README.md \
  --model-card-artifact /path/model-card.evidence.json \
  --pii-bench-report /path/seed97-pii-bench-zh-posthoc.json \
  --final-model-manifest /path/final-model-binding.json \
  --community-v2-preauthorization \
  --output-dir /path/model-package
```

模型包完成后，许可证报告必须从该模型包自己的三份法律文档派生，不能再引用仓库根目录的旧 NOTICE：

```bash
PYTHONPATH=src python scripts/produce_community_cascade_release_v2_artifacts.py license-report \
  --sbom-artifact /path/sbom.json \
  --model-package-root /path/model-package \
  --output /path/license-report.json
```

```bash
PYTHONPATH=src python scripts/build_community_cascade_release_v2_contract.py \
  --quality-receipt /path/open24-quality-result-v2.json \
  --quality-replay-receipt /path/quality-strict-replay.json \
  --quality-native-replay /path/quality-strict-native-replay.json \
  --final-model-manifest /path/final-model-binding.json \
  --service-configuration-manifest /path/service-configuration-binding.json \
  --pii-bench-posthoc-report /path/seed97-pii-bench-zh-posthoc.json \
  --calibration-bundle /path/calibration.json \
  --service-source-manifest /path/community-service-source-identity.v4.json \
  --predecessor-service-source-manifest /path/community-service-source-identity.v3.json \
  --internal-preopen-unlock /path/release-eval-v2-internal-unlock.json \
  --candidate-manifest model_raw=/path/model-raw.candidate-provenance.json \
  --candidate-manifest full_system=/path/full-system.candidate-provenance.json \
  --comparator-manifest model_raw=/path/model-raw.comparator-provenance.json \
  --comparator-manifest full_system=/path/full-system.comparator-provenance.json \
  --performance-manifest model_raw=/path/model-raw.performance.json \
  --performance-manifest full_system=/path/full-system.performance.json \
  --replay-receipt model_raw:candidate_prediction_full_replay=/path/model-raw.candidate-replay.json \
  --replay-receipt model_raw:comparator_generation_replay=/path/model-raw.comparator-replay.json \
  --replay-receipt model_raw:performance_harness_replay=/path/model-raw.performance-replay.json \
  --replay-receipt full_system:candidate_prediction_full_replay=/path/full-system.candidate-replay.json \
  --replay-receipt full_system:comparator_generation_replay=/path/full-system.comparator-replay.json \
  --replay-receipt full_system:performance_harness_replay=/path/full-system.performance-replay.json \
  --native-replay-evidence model_raw:candidate_prediction_full_replay=/path/model-raw.candidate-native-replay.json \
  --native-replay-evidence model_raw:comparator_generation_replay=/path/model-raw.comparator.execution-replay.json \
  --native-replay-evidence full_system:candidate_prediction_full_replay=/path/full-system.candidate-native-replay.json \
  --native-replay-evidence full_system:comparator_generation_replay=/path/full-system.comparator.execution-replay.json \
  --artifact final_model_manifest=/path/final-model-binding.json \
  --artifact model_card=/path/model-card.evidence.json \
  --artifact model_package_manifest=/path/model-package-manifest.json \
  --artifact service_configuration_manifest=/path/service-configuration-binding.json \
  --artifact service_source_manifest=/path/service-source-identity.json \
  --artifact technical_documentation_manifest=/path/technical-docs-manifest.json \
  --artifact wheel_manifest=/path/wheel-manifest.json \
  --artifact wheelhouse_manifest=/path/wheelhouse-manifest.json \
  --artifact container_manifest=/path/container-manifest.json \
  --artifact sbom=/path/sbom.json \
  --artifact license_report=/path/license-report.json \
  --artifact benchmark_report=/path/open24-quality-result-v2.json \
  --artifact public_artifact_scan=/path/public-artifact-scan.json \
  --artifact dependency_scan=/path/dependency-scan.json \
  --verification unit_tests=/path/unit-tests.json \
  --verification clean_wheel_smoke=/path/clean-wheel-smoke.json \
  --verification container_smoke=/path/container-smoke.json \
  --verification offline_model_smoke=/path/offline-model-smoke.json \
  --verification offline_service_smoke=/path/offline-service-smoke.json \
  --model-package-root /path/model-package \
  --release-wheel /path/pii_zh_qwen-0.2.0rc1-py3-none-any.whl \
  --wheelhouse-root /path/offline-wheelhouse \
  --output /path/community-cascade-release-v2.completed.json
```

builder 成功后，可将同一组 evidence 参数传给
`validate_community_cascade_release_v2.py`，额外指定
`--contract /path/community-cascade-release-v2.completed.json`，并把 `--output` 改为新的 release receipt
路径。先不带 `--output` 做确定性的本地源闭包 preflight（会逐字节哈希 release model package，
但不联网、不加载权重推理）；只有 stdout 状态精确为
`READY_FOR_USER_AUTHORIZATION` 才生成本地 release receipt。

服务 source manifest 也不能手填。正式生成入口是：

```bash
PYTHONPATH=src python scripts/produce_community_cascade_release_v2_evidence.py service-source \
  --final-model-manifest /path/final-model-binding.json \
  --service-configuration-manifest /path/service-configuration-binding.json \
  --full-system-candidate-manifest /path/full-system.candidate-provenance.json \
  --predecessor-source-manifest /path/community-service-source-identity.v3.json \
  --output /path/community-service-source-identity.v4.json
```

`--predecessor-source-manifest` 必须是普通、非 symlink、精确 mode `0444` 的 v3 文件。producer 会独立验证
v3 closed schema、自哈希、文件 hash、完整 runtime inventory 及其与同一 model/service/candidate 链的一致性，
再从当前树重建 v4 全量 inventory；caller 不能手填 predecessor 或 delta。

`quality-replay` 和组合式 `candidate-replay` 同时接收 `--native-output` 与 `--output`；前者保存实际 replay
sidecar，后者保存 gate wrapper。若 quality 尚未结束，可先用 `candidate-native-replay` 做真实模型重放，
它只生成 native sidecar；quality 完成后再用 `candidate-wrapper --native-replay ...` 绑定 quality 并生成
wrapper。native PASS 没有可由外部 caller 直接调用的 receipt builder：producer 只会在 generator 子进程、
预测与 generation receipt 字节比较、provenance strict replay 全部成功后发布 sidecar。
候选 sidecar 还会绑定 replay 前后均未变化的完整 runtime package inventory；wrapper 与最终 gate 都要求该
inventory 等于 v4 source identity，因此源码漂移后的旧 sidecar 不能与新 source manifest 拼接。
`comparator-replay`
通过 `--execution-replay` 消费既有 native receipt 并只生成 wrapper。candidate 两轨必须串行执行；执行前
先只读检查 GPU 并用单个 `CUDA_VISIBLE_DEVICES` 隔离空闲卡，不得杀进程或抢占其他任务。

quality strict replay receipt 通过 `--quality-replay-receipt` 单独传入，因此上面的六个 track receipt 加上
它正好是七份。只有七份 exact replay、所有 identity、artifact、verification 全部通过时，状态才可能
成为 `READY_FOR_USER_AUTHORIZATION`。community-v2 的 schema 会拒绝任何 `authorized=true`、
`published=true`、target ID 或 publication receipt；GitHub/Hugging Face 发布必须由用户另行明确授权并由
未来的独立 successor 合同完成，本门禁不会自动发布，也不能被改写成 `PUBLISHED`。

## 声明边界

即使门禁通过，也只允许在具名、冻结、公开测试已见的 Open-24 公开/合成协议中描述结果。它不能证明
真实世界泛化、生产可用、“全球 SOTA”或“首个中文 PII 模型”，也不会解除任何历史 production gate。
