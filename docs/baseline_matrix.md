# 中文 PII 同协议基线矩阵

状态：**执行中；不得在 hidden test 完成前声称“最佳”**

本文冻结开源模型、公开 checkpoint 与开源框架的比较口径。目标不是把不同标签集的自报分数
拼成排行榜，而是在同一 gold、同一字符 offset、同一解码和同一误报约束下独立复现。

Comparator registry Generation 1 已撤回。其声明的 `08:39:42` 冻结时间早于它绑定的 schema、
Argus 报告和 registry 文件本身，当前字节不能证明该时间点的预注册。原文件保持不变以便审计，正式记录见
[`comparator_registry_v1_withdrawal.yaml`](../configs/evaluation/comparator_registry_v1_withdrawal.yaml)。
独立的 OpenMed Generation-2 v1 随后已按真实时间冻结，但三次执行都在完整可比较 aggregate 产生前
fail closed：Snowflake 首批失败；QwenMed 写出两份私有 Formal prediction 后在 Chat 首批失败；
Privacy Filter 在运行时隔离门失败，因此 BigMed 未启动。v1 八轨比较已撤回，公开可比较结果仍为
`0/8`。其后通过独立 repair finalization（文件/逻辑 SHA-256
`58171cdc…` / `c1e8a4e1…`）和授权 umbrella（`1ad600d7…` / `57415577…`）完成四路
evaluator gate，并在全新目录完成 OpenMed 四模型 × `model_raw`/`framework_hybrid` 的 v2 `8/8`
重跑。所有新结果只标记为 `post-freeze implementation-repair public benchmark`，且
`first=false`、`blind=false`、`claim_eligible=false`。Hidden gold 仍未物化，sealed zero-access
凭证、access ledger 与签核尚未建立；所有“best”声明继续禁用。

## 当前 repository-wide Generation-2 comparator candidate freeze

Repository-wide Generation-2 comparator candidate set 已于 `2026-07-16T04:26:44+08:00`
按真实时间冻结为 `frozen_candidate_set_blocked`。它固定 12 systems / 18 track
variants，其中 9 条 legacy 和 8 条 OpenMed 共 17 条是已见公开合成 benchmark
上的描述性诊断轨；第 18 条是新近公开、支持中文但尚未在仓库内闭包的
PII Engineer Multi NER v2.1 必需 pending 轨。这是 blocked candidate-set freeze，不是
active leaderboard 或 claim gate。

该第 18 条轨已在 freeze **之后**完成 result-blind preregistration 和同协议 CPU 复现；这是一份
additive、non-activating 证据，不能回写下面的 immutable registry/receipt，也不会把历史
`6 PASS / 4 BLOCKED` 改成 `6 PASS / 3 BLOCKED`。其技术复现已闭合，但上游模型卡
Apache frontmatter、AGPL body 与 GitHub Apache 源码之间的权重许可范围仍未解决，因此发布和
再分发继续 `BLOCKED`。

Blocked receipt 记录 `6 PASS / 4 BLOCKED`。PASS 项是 candidate-set identity、Generation-1
withdrawal chain、public-search inventory、closed-8 public diagnostics、source/mapping/runner bindings
和 claims-disabled policy。四个不能省略的 blocker 是：

1. `pii_engineer_multi_same_protocol_reproduction`：冻结当时缺少 PII Engineer Multi 的仓库内
   artifact、Apache-frontmatter/AGPL-body 许可矛盾审查、mapping、runner 和同协议结果；
2. `open24_and_human_hidden_mapping_closure`：Open-24 成员级 mapping 与 zero-access human hidden
   尚未物化；
3. `three_metric_paired_bootstrap_implementation`：冻结时的 bootstrap 只支持 strict micro F1，
   当时尚缺 strict macro F1 与 PII-free document FPR；
4. `legacy_hybrid_semantics_source_closure`：ERNIE/CLUENER 历史 hybrid 是 `LocalRecognizer` +
   custom fusion，不是 release-native `AnalyzerEngine` 语义。

第 4 项在 freeze 后已有 additive technical closure：Presidio AnalyzerEngine `2.2.363`
CPU-only v2 用分开注册的 `BioTokenClassifierRecognizer` 与 `CnCommonRecognizer` 完成两条
release-native `framework_hybrid` 复现，并冻结 aggregate、result closure 与 receipt。它不会原地
改写上述 registry 的 `6 PASS / 4 BLOCKED`，也没有解决权重许可、human hidden、PII-free FPR
或 claims；successor active registry 仍是激活前置条件。

所有 `first`、`blind`、`best`、`global_best`、`SOTA` 和 registry claim eligibility
均为 false；只允许在已披露限制内做 descriptive reporting。当前冻结不允许 human-hidden
执行或 claim activation，也不得原地修改；四项关闭后必须生成 successor active registry。

内容寻址证据为：

- [schema](../configs/evaluation/comparator_registry_v2_candidate.schema.json) 文件 SHA-256：`8daf14beaaf88a70ff49bd229b322e76714518ea939cb51b35d3f837c80336b4`；
- [builder](../scripts/build_comparator_registry_v2_candidate.py) 文件 SHA-256：`6d621fee09fb70e136ba3d24607511da53671b951d3efab54b40000c535f1f0d`；
- [blocked candidate registry](../configs/evaluation/comparator_registry_v2_generation_2_candidate.json) 文件/canonical SHA-256：`42ee4a48984bca02a5a6ddaac3af01f0ea5bb488020a324b153e9fdfcd6bc221` / `fc36c2e070e359227061bb67ae0b527053126ada6726a292c2dc766add6c2baa`；
- [blocked receipt](../reports/evaluation/comparator_registry_v2_generation_2_candidate_blocked.json) 文件/canonical SHA-256：`3fa5d9527e102be46d32856fc24c9c6d5365907de1005f710f3f66eb8d6f9c1d` / `e60b2298247efc10ce0a360c65971f93bf43d5b78553f550f4345abd860d2217`。

Strict-local replay 要求精确 bytes、policy、schema、`0444` mode 和原始 mtime。Portable
Git-checkout replay 可接受 checkout 后的 `0644`，但必须同时提供 registry 与 receipt
的外部 file SHA-256 锚，且不声称重建原始 mtime。这两枚锚必须来自 release
commit/tag 或本文已记录的值，不得从待验文件现算再回填。
`chinese_common_ner` 不贡献这 12 个 systems 或 18 条 tracks，继续为
reference-only/quarantine，不进入 D1、训练、正式/hidden 评测、模型权重或候选 ancestry。

### Freeze 后的三指标实现组件

版本化的 successor 实现现已覆盖 strict micro F1、固定完整有序标签集（包含当前 scope 中
zero-support 标签）的 strict macro F1，以及只以显式 eligibility manifest 为分母的 PII-free
document FPR。这只关闭 implementation-only 子组件，不改写上述 immutable receipt；旧状态仍是
`6 PASS / 4 BLOCKED`，不是 `6 PASS / 3 BLOCKED`。

正式重采样输入必须由互斥、与模型结果无关的 atomic partition 构成；reporting slices 可以重叠，
但共享同一个 document-paired resample plan，pooled 中每个唯一文档只计一次。协议 gold 为空不能
自动推断为 PII-free，OOD-only 或 mapping 后为空的正例也不能进入 FPR 分母。F1 的
`candidate - comparator` 95% CI 下界必须严格大于 0；FPR 要求候选点估计不超过 5%，且
`comparator + 0.005 - candidate` 的 95% CI 下界不小于 0。

当前实现故意固定输出 formal claim gate 为 false。未来 successor active registry 还必须在 hidden
解封前绑定 Open-24、候选与精确 comparator universe、完整 eligibility manifest、允许的 reporting
slice 全集、与 outcome 无关的 atomic-partition manifest、每个 scope 的 required metrics 及其内容
哈希，并关闭其余三个历史 blocker；否则不得执行 human hidden 或激活任何“最佳”声明。

本 implementation-only 闭包于 `2026-07-16T05:35:00+08:00` 冻结，正式只读文件为：

- `configs/evaluation/three_metric_paired_bootstrap_v2_preregistration.freeze.json`（内部冻结件，首发不公开）：file SHA-256 `1a083205fe7b6bcf4a0605ae6c149f6981ba67335f635b4fda0c8a63deffd252`，canonical SHA-256 `7cede697c0b0fb4b92ea9153490a9f6fd73923d4468a8f984fb6aa8d55b4e09a`；
- [three-metric v2 implementation closure receipt](../reports/evaluation/three_metric_paired_bootstrap_v2_implementation_closure.json)：file SHA-256 `297a83b6546efc329e832c76cf6f55010a155c05adf3cf9d09cc3a1c682075ad`，canonical SHA-256 `9343fdce818bf96d59d3faf6c3ea184998e9fdb92acb7b42c9a023f107ae1bee`。

两份文件 mode 均为 `0444`，状态均为
`IMPLEMENTATION_COMPONENT_CLOSED_NON_ACTIVATING`。Receipt 记录 `hidden_gold_read=0`，并将
human-hidden execution、registry activation、claim activation 以及
`first/blind/best/global_best/sota` 全部固定为 false。闭包未读取 benchmark rows 或 predictions，
也未加载模型权重；因此这些 PASS check 只证明实现与元数据绑定自洽，不是模型/级联效果结果。

严格 replay 校验 exact bytes、policy、schema、`0444` mode 和原始 mtime：

```bash
python scripts/build_three_metric_paired_bootstrap_v2_closure.py --verify-existing
```

Portable Git-checkout replay 允许 `0644`，但必须由可信 release commit/tag 或上述记录提供两枚
外部 file SHA-256 锚，不能从待验文件现算再回填：

```bash
python scripts/build_three_metric_paired_bootstrap_v2_closure.py \
  --verify-existing \
  --portable-git-checkout \
  --expected-freeze-file-sha256 1a083205fe7b6bcf4a0605ae6c149f6981ba67335f635b4fda0c8a63deffd252 \
  --expected-receipt-file-sha256 297a83b6546efc329e832c76cf6f55010a155c05adf3cf9d09cc3a1c682075ad
```

Replay 成功不改变旧 Generation-2 `6 PASS / 4 BLOCKED`，不授权读取 human hidden，也不激活
正式 leaderboard 或 claim gate。

独立对抗审计发现，原 implementation pair 本身没有逐一绑定正常导入路径上的全部项目内依赖；
例如 `evaluation/io.py` 被改写后，原 pair 仍可能 replay 成功。原 pair 保持原字节，新增 additive
dependency pair 专门关闭这一缺口：

- `configs/evaluation/three_metric_paired_bootstrap_v2_dependency_closure.freeze.json`（内部冻结件，首发不公开）：file SHA-256 `37e817e55a5f0e8775de686ad5bfc8a219ec3231d1c7a3a3c577de62debc5eae`，canonical SHA-256 `7a128f289b7ed0d2f7e3e3a635c1f10509a2677df6a4f25099838ec086bb3155`；
- [dependency closure receipt](../reports/evaluation/three_metric_paired_bootstrap_v2_dependency_closure.json)：file SHA-256 `e3cf6f5114757410b0a4eb18210e4e29807a60f35e0818c9c69af68b35d1be9c`，canonical SHA-256 `fb972648413cd1a58a20ac4f30a6a1e818dc302a12966416bd1cb8d57954de2f`。

两份文件冻结于 `2026-07-16T06:40:00+08:00`，mode 均为 `0444`，状态为
`DEPENDENCY_CLOSURE_CLOSED_NON_ACTIVATING`。它们绑定隔离 `python -B -S` 正常导入得到的精确 8 个
项目内模块、全部 support inputs、旧 pair 和 Generation-2 `6 PASS / 4 BLOCKED`。首发 generation
还要求独立审计者在构建器外提供 builder/schema/tests raw SHA-256：
`7780cfb23e11e521a86a93a622b5b195e92028b786a71ff1ee2498fb362d5f49`、
`204094357b7258d98572a20da83003ee07bc83b2a4e10188e2ad93d772d29bf5`、
`09152dc9485590db2110a84a087d878ee31b407be466cc3ba56309c63fa65020`。缺失或错配会在输出前
fail closed；构建器若已被替换成忽略参数的版本则无法自证，因此这三枚值必须外部核对。

严格 replay：

```bash
python scripts/build_three_metric_paired_bootstrap_v2_dependency_closure.py --verify-existing
```

Portable replay：

```bash
python scripts/build_three_metric_paired_bootstrap_v2_dependency_closure.py \
  --verify-existing \
  --portable-git-checkout \
  --expected-freeze-file-sha256 37e817e55a5f0e8775de686ad5bfc8a219ec3231d1c7a3a3c577de62debc5eae \
  --expected-receipt-file-sha256 e3cf6f5114757410b0a4eb18210e4e29807a60f35e0818c9c69af68b35d1be9c
```

两层 closure 合起来只关闭 implementation/runtime-dependency 组件，仍不授权 human hidden 或
claims。新增实现文件晚于历史 OpenMed source closure；旧 exact-source verifier 在当前树上必须对
这一枚已登记 extra fail closed。历史冻结结果不变，新的 OpenMed 正式重跑必须新建 v3 successor
source closure/gates，不能放宽旧 verifier。

## 1. 两个主协议

### Closed-8：与 PII Bench ZH 对齐

只评估 `PERSON_NAME`、`PHONE_NUMBER`、`CN_RESIDENT_ID`、`ADDRESS`、`EMAIL_ADDRESS`、
`BANK_CARD_NUMBER`、`PASSPORT_NUMBER`、`VEHICLE_LICENSE_PLATE`。第三方模型不属于这八类的
logit 在解码前屏蔽；同义来源标签先按冻结映射归一化。模型没有覆盖的协议标签仍计 FN，且必须
同时报告 `covered_labels / 8`，避免两类 NER 模型以较小任务获得虚高结论。

### Open-24：项目真实产品边界

在未来独立物化并封存、且候选冻结前从未查看的 hidden test 上评估 taxonomy core 24 类。未支持标签
计 FN；输出错误类型计 FP。
这是决定能否声称项目模型优于现有公开方案的主要协议。PII Bench ZH 只能作为外部合成压力集，
不能替代 human-authored-context hidden test。

两个协议都采用 exact `(start, end, label)` character-span micro F1；另报 macro F1、per-label
precision/recall/F1、PII-free 文档 FPR、每千字符误报数、吞吐和峰值显存。阈值只能用 validation
冻结，禁止查看 test 后调参。

## 2. 系统轨道必须拆开

| 轨道 | 允许组件 | 回答的问题 |
|---|---|---|
| `model_raw` | checkpoint + 固定 decoder | 独立模型学到了什么 |
| `model_calibrated` | model + validation-only 阈值/温度 | 公平校准后模型表现 |
| `rules_only` | Presidio/中文规则与 validator，不调用 NER | 确定性框架基线 |
| `framework_hybrid` | Presidio + 指定公开 NER + 同一规则包 | 常见工程组合基线 |
| `full_system` | 本项目模型 + rules + fusion + calibration | 可部署方案的整体表现 |

模型分数不得写成 full-system 分数；framework hybrid 也不得与 model-only 混排而不标注轨道。

## 3. 首批冻结对象

下表是人类可读的历史/项目规划清单，不是 machine registry 的完整 universe；
当前 12-system 候选集、轨道和状态以上述 blocked candidate registry 为准。

| ID | 组件 | 固定 revision | 标签覆盖与限制 | 状态 |
|---|---|---|---|---|
| `aiguard_qwen3_0_6b` | `ZJUICSR/AIguard-pii-detection-fast` | `677a5ebc1600fef61e8973cafd3026be322b3a73` | 中文 PII 21 类；Apache-2.0 | Closed-8 已复现 |
| `openai_privacy_filter` | `openai/privacy-filter` | `7ffa9a043d54d1be65afb281eddf0ffbe629385b` | 英语为主；5/8 语义覆盖；Apache-2.0；官方 constrained Viterbi | Closed-8 已复现 |
| `presidio_cn_rules` | Presidio Analyzer + 本仓 `cn_common` | 代码与规则 SHA | 结构化标识强、不支持姓名/地址 | Closed-8 已复现 |
| `argus_redact_fast` | `argus-redact` fast（无 NER/LLM） | `0.7.18` / `88d0342012b43e6c71fe79a02b90a1432db0f9a9` | 8/8；Apache-2.0；与 PII Bench ZH 同作者且含其生成器/回归测试 | Closed-8 已复现；开发耦合 |
| `presidio_ernie_ner` | Presidio + `gyr66/Ernie-3.0-base-chinese-finetuned-ner` | `3f3c3fa40cf87298586dbbed55fd7ca1988719a8` | name/address/mobile/email/QQ；模型卡未声明 license，不再分发权重 | Model-only、历史 custom fusion 与 release-native v2 均已复现；claims BLOCKED |
| `presidio_cluener` | Presidio + `uer/roberta-base-finetuned-cluener2020-chinese` | `cddd8fc233e373855a8c0a7f4b7eb83acb686a2b` | closed-8 只直接覆盖 name/address；模型卡未声明 license | Model-only、历史 custom fusion 与 release-native v2 均已复现；claims BLOCKED |
| `pii_engineer_chinese_ner` | `pii-engineer/PII-Engineer-Chinese-NER-v1.0` | `d16aa1c54f34f993b2d4c30c218ee9d712959b2d` | 实际为 BERT BIO ONNX；4/8 标签；上游 Apache/AGPL 声明冲突，不再分发权重 | Closed-8 已复现 |
| `pii_engineer_multi_ner_v2_1` | `pii-engineer/PII-Engineer-Multi-NER-v2.1` | `8ce12a2da5c333483ca2bd9a85def9d60609f244` | multilingual/zh；7/8，bank card unsupported；Apache frontmatter 与 AGPL body 冲突 | Post-freeze result-blind 同协议 CPU 复现完成；license/claims 仍 BLOCKED |
| `project_qwen_model` | 本项目 checkpoint | training manifest + weight SHA | closed-8 与 open-24 | 历史 v1/v6 仅作诊断；v6 无可行 checkpoint，v7 尚未训练且无 selected candidate |
| `project_full_system` | 本项目 checkpoint + rules/fusion/calibration | 全部 artifact SHA | closed-8 与 open-24 | 历史 v1 已诊断；v7 candidate、profile 与 human hidden 均未冻结 |

### 本项目独立执行的结果

下表只排列使用相同 PII Bench ZH closed-8 gold、解码前标签约束和 exact character-span
evaluator 的独立重跑。`model_raw`、`rules_only`、`framework_hybrid` 是不同赛道，不能用
hybrid 分数冒充某个模型 checkpoint 的分数。PII Bench ZH 没有 PII-free 文档，因此本表也
不能回答真实部署误报率。

| 对象 | 赛道 | 覆盖 | Formal F1 | Chat F1 | Pooled F1 | 结论 |
|---|---|---:|---:|---:|---:|---|
| 项目 v1 selected | `model_raw` | 8/8 | 0.5580 | 0.3890 | 0.4992 | 全合成训练；明显落后 |
| 项目 no-class-weight seed 42 | `model_raw`（JPT 诊断） | 8/8 | 0.6090 | 0.4786 | 0.5669 | 证明强 class weight 有害，但仍不够 |
| ERNIE 中文 NER | `model_raw` | 4/8 | 0.5790 | 0.5301 | 0.5631 | 其余协议标签计 FN |
| PII Engineer 中文 NER | `model_raw` | 4/8 | 0.7183 | **0.7073** | 0.7148 | 撤回 generation-1 的 Chat 诊断包络；artifact/模型卡与许可证均有冲突 |
| PII Engineer Multi NER v2.1 | `model_raw` | 7/8 | 0.3897 | 0.3787 | 0.3860 | Result-blind prereg 后 CPU 复现；raw pooled 在本表末位，license/claims BLOCKED |
| AIguard Qwen3-0.6B | `model_raw` | 8/8 | **0.8014** | 0.5714 | **0.7364** | 撤回 generation-1 的 Formal/Pooled model-only 诊断包络 |
| OpenAI Privacy Filter | `model_raw` | 5/8 | 0.5840 | 0.4005 | 0.5306 | 官方 Viterbi/default OP；英语主模型；三类 unsupported 计 FN |
| CLUENER2020 RoBERTa | `model_raw` | 2/8 | 0.5753 | 0.5735 | 0.5747 | 姓名/地址强，其余 6 类计 FN；本项目无可验证再分发授权 |
| `cn_common_v5` | `rules_only` | 6/8 | 0.6780 | 0.6563 | 0.6710 | 高精度、低覆盖；不支持姓名/地址 |
| `argus-redact 0.7.18` fast | `rules_only` | 8/8 | **0.9355** | **0.9107** | **0.9275** | 撤回 generation-1 的规则轨诊断包络；与 benchmark 同作者，不能视为独立外测 |
| 历史 Presidio + ERNIE + `cn_common_v5` | `framework_hybrid`（custom fusion） | 8/8 | 0.9043 | 0.8786 | 0.8960 | `LocalRecognizer` + 外层融合；不是 stock/release-native `AnalyzerEngine` |
| 历史 Presidio + CLUENER + `cn_common_v5` | `framework_hybrid`（custom fusion） | 8/8 | 0.9527 | 0.9361 | 0.9474 | 撤回 generation-1 诊断包络；不是 stock/release-native `AnalyzerEngine` |
| Presidio 2.2.363 native + ERNIE | `framework_hybrid`（release-native v2） | 8/8 | 0.90517850 | 0.87925378 | 0.89680396 | CPU-only；pooled macro 0.83138344；result closed、non-activating |
| Presidio 2.2.363 native + CLUENER | `framework_hybrid`（release-native v2） | 8/8 | **0.95273279** | **0.93613793** | **0.94736372** | CPU-only；pooled macro 0.86673384；result closed、license/claims BLOCKED |

### Release-native Presidio AnalyzerEngine v2 复现

表中两组 Presidio 行不是同一个 runtime 的重复展示。历史 `0.89599034 / 0.94736372`
pooled micro F1 来自 `LocalRecognizer` 后再做外层 custom fusion；release-native v2 则由
Presidio AnalyzerEngine `2.2.363` 分开注册 `BioTokenClassifierRecognizer` 与
`CnCommonRecognizer`，由 AnalyzerEngine 自己执行规则/NER 去重和 overlap 语义。v2 固定
offline、CPU-only、相同 closed-8 gold、无 benchmark 阈值调优，结果为：

| Native v2 track | Formal micro / macro | Chat micro / macro | Pooled micro / macro | Pooled TP / FP / FN |
|---|---:|---:|---:|---:|
| ERNIE | 0.90517850 / 0.84294878 | 0.87925378 / 0.78701966 | 0.89680396 / 0.83138344 | 21,578 / 3,338 / 1,628 |
| CLUENER | 0.95273279 / 0.87321090 | 0.93613793 / 0.83997016 | 0.94736372 / 0.86673384 | 21,229 / 382 / 1,977 |

v1 在四份私有预测完成后、公开 aggregate 写入前，因 validator 先舍入每类 F1 再求 macro
导致第八位小数差一而 fail closed；它已标记
`WITHDRAWN_VALIDATION_FAILURE_SUPERSEDED`，且不允许引用 v1 指标。根因和未发布状态由
[v1 failure receipt](../reports/evaluation/native_presidio_zh_ner_framework_hybrid_v1_failure_receipt.json)
保存。修正为“从 TP/FP/FN 计算未舍入 per-label ratio，求均值后只舍入一次”后重新冻结 v2，
没有把修复后的代码冒充原 v1。

v2 的只读证据链包括
[preregistration](../configs/evaluation/native_presidio_zh_ner_framework_hybrid_v2.freeze.json)、
[aggregate](../reports/baselines/native_presidio_zh_ner_framework_hybrid_v2.json)、
[result closure](../configs/evaluation/native_presidio_zh_ner_framework_hybrid_v2_result_closure.freeze.json)
与 [closure receipt](../reports/evaluation/native_presidio_zh_ner_framework_hybrid_v2_result_closure_receipt.json)。
它们的 file/canonical SHA-256 分别为 `979a3d83…f87e5 / fa15dda0…cf80`、
`dbf55c58…a720c / a4308adf…cf69`、`b64b77d5…716f / 595e612b…49c`、
`17282045…c8 / 499bbe9c…15d`。Closure 状态是
`RESULT_CLOSED_NON_ACTIVATING / BLOCKED_PUBLIC_TEST_EXPOSED`。

这 8,000 篇 PII Bench ZH 文档是已公开、已暴露、程序化合成数据，而且没有 PII-free 文档，
所以 document FPR 不可估计；结果既不是 blind/human-hidden，也不能用于继续调规则或路由。
ERNIE 与 CLUENER 权重许可均未闭合，`redistribution_allowed=false`、
`release_bundle_allowed=false`。所有 first/blind/best/global-best/SOTA 与 registry/claim activation
开关保持 false；v2 只关闭 native runtime 的技术复现问题，不证明生产服务最佳。

### 第一版 CLUENER 完整服务回归硬门

用户提供的第一版结果把旧 Presidio + 中文 ERNIE + 自定义规则服务报告在公开 CLUENER dev。
内部第一版证据审计 `docs/first_version_baseline_audit.md` 固定了 1,343 篇、10 类、3,072 个展开的 span
occurrence，以及旧服务 `TP/FP/FN=2525/820/547`。发布比较不能使用显示到四位小数的
`0.7870`，必须使用精确历史下限：

```text
historical_micro_f1 = 2*2525 / (2*2525 + 820 + 547) = 5050/6417
```

内部回归门 `docs/legacy_cluener_regression_gate.md` 当前是
`G0 PASS / G1 REPLAY_PASS_HISTORICAL_ATTRIBUTION_UNRESOLVED / G2-v2 FAIL / G2-v3 LIMITED_COMPATIBILITY_ELIGIBLE_NONACTIVATING`。
G1 的 1,343 行 CPU/offline replay 精确得到 `2525/820/547`、F1 `5050/6417`，与 DOCX 整数
相等；公开 receipt 绑定了幸存旧仓/runtime/私有预测，但 DOCX 未绑定该执行身份。首次 G2-v2
actual service 得到 `2528/828/544`、F1 `5056/6428`，严格低于 G1；其 paired ΔF1 为
`-0.000413300118`、95% CI `[-0.001331198733,0.000524068703]`，失败史由
[v2 closure](../reports/evaluation/legacy_cluener_g1_g2_comparison_v1.json) 保留。

在 v2 结果披露后，G2-v3 只把同一实际服务路径的共享路由阈值改为 `0.50`，重新运行得到
`2510/736/562`、F1 `5020/6318=0.794555238999…`。整数比较为
`32213340 > 31905900`，严格超过 G1；[paired successor closure](../reports/evaluation/legacy_cluener_g1_g2_threshold_050_v3_comparison_v2.json)
状态为 `CLOSED_LIMITED_COMPATIBILITY_REGRESSION_ELIGIBLE_NONACTIVATING`，ΔF1
`+0.007583133655`、描述性 95% CI `[0.004883808071,0.010453811188]`、1,000/1,000 次胜出。
阈值选择明确是 post-result/nonblind，不能证明未见数据泛化。单独 NER 只做消融和贡献解释，
不阻断服务发布。

这 10 类包含 `book/game/movie/scene` 等非 PII 语义类，因此只允许显式启用非默认
`legacy_cluener_10` compatibility profile；默认生产 profile 仍是 `pii_core`。G2-v3 虽已通过，
也只能写“post-result/nonblind 的实际服务在公开、test-exposed CLUENER dev 兼容回归上超过
第一版/G1”，不能把该数字当作生产 PII 效果、hidden test 或最佳声明，且不能替代 selected/default、
结构化 PII、负例 FPR、延迟、GPU/container、license、release 和 human-hidden 门。

### PII Engineer Multi v2.1 post-freeze 复现

该轨在读取 benchmark rows 或 predictions 之前冻结了模型 revision、十文件模型布局、官方源码
commit、Rust runner 与二进制、ONNX Runtime 1.20 CPU dylib、closed-8 mapping、官方默认阈值和
数据 manifest。只读 [preregistration](../configs/evaluation/pii_engineer_multi_v2_1_closed8_model_raw_v1.freeze.json)
的 file/canonical SHA-256 分别为
`3e6d8abe7a29756d12a144a4e62e8a8e350e1e533cac7f507f508fcd787f2af8` /
`f28da6464b0feb4184d111838c7ccd1b446d2c43ede0638397ff5b321439d651`；strict replay 与一个正常
`0644` checkout 副本的 portable replay 均通过。8,000 条 CPU 执行耗时 446.33 秒，公开
[aggregate report](../reports/baselines/pii_engineer_multi_ner_v2_1_pii_bench_zh.json) SHA-256 为
`ecf064cec8ad3346a316f15e7ba6fac5fdbff5def253b9fded25852e8023068d`。

结果随后由 additive、non-activating 的只读
[result closure](../configs/evaluation/pii_engineer_multi_v2_1_closed8_model_raw_v1_result_closure.freeze.json)
与 [receipt](../reports/evaluation/pii_engineer_multi_v2_1_closed8_model_raw_v1_result_closure_receipt.json)
绑定。两者 file/canonical SHA-256 分别为
`56ccd6e15d41a3727c84f23816906a51fa3700634360aa18b3defa8e29494550` /
`8f1ffa521c241103df1323769eb996d57ec0ae646937736ff4fbfe3fe670d8fe` 和
`93c295286f66c91cbccccbc699619c6d830ff631dc9a43130f886afb3c550027` /
`142e44f67e81e08a5d7a21b2c6ebd2bf65be9983ee3df0e3f81391881bd837e1`；strict replay 与普通
`0644` checkout 副本的 portable replay 均通过。Closure 只读取公开 aggregate/preregistration，
不读取 benchmark rows、私有 predictions、hidden gold 或权重；只保存两份私有 prediction 的哈希，
不保存路径或逐条内容。其 `overall_status=BLOCKED`，不会激活旧 registry、human hidden、许可、
再分发或任何 claim。

独立审计为 `P0=0 / P1=0 / P2=1 / ACCEPT_WITH_P2`。唯一 P2 是发布失败措辞偏强：当前实现是
freeze 后 receipt 的有序 no-clobber 发布；若最后的临时硬链接清理本身报错，完整 canonical pair
与 orphan temp links 可同时保留。因此现有 pair 的内容和 non-activating 状态有效，但它不能被解释为
所有文件系统故障下严格 all-or-nothing；后续 successor 需修复 cleanup fault handling 或收窄该声明。

Formal/Chat/Pooled strict micro F1 为 `0.38965434 / 0.37870855 / 0.38601245`，pooled strict macro
F1 为 `0.29449612`。Pooled precision `0.76774028`，但 recall 只有 `0.25782125`；bank card 不受
支持，bank card 与 passport 均为零预测。Address、phone、plate 的 pooled precision 分别为
`0.99122807 / 0.92303897 / 0.98260870`，对应 recall 只有
`0.33551069 / 0.28088876 / 0.16282421`，且 Chat 的 address/plate exact F1 都为 0。它略低于
BigMed raw pooled micro `0.38644388`，低于本表所有其他 `model_raw` pooled 结果，也显著低于
`cn_common_v5` rules-only `0.6710` 与两个历史 framework-hybrid 结果。

这不证明模型在真实中文文本中无价值：PII Bench 是已见、100% 程序化合成数据，而且没有一篇
可用于 document FPR 的 PII-free 文档。它只说明不能把该 checkpoint 单独宣传成更强方案，也不能
依据这次公开结果事后调 cascade 路由。若上游先解决权重许可，最多可在独立 D1 dev 上预注册验证其
高精度低召回分支是否带来 unique TP，再在未见 human hidden 一次性确认；未经该步骤，不进入默认
服务 profile。上游 tokenizer 的 512 总序列右截断还会令未编码尾词保持零向量但继续参与 span
scoring，长文限制不能写成简单“尾部丢弃”。所有 first/blind/best/SOTA、registry activation、
权重再分发和 release bundle 开关保持 false。

### OpenMed Generation-2 post-freeze repair 结果

下表来自 v1 撤回后、经共同 repair gate 授权、在全新 run 目录执行的 v2 完整 `8/8`。它保留
Formal/Chat 分套结果，同时只在 pooled 列补充 macro F1；不得把 hybrid 分数归因于模型权重。

| OpenMed 对象 | 赛道 | Formal micro F1 | Chat micro F1 | Pooled micro F1 | Pooled macro F1 |
|---|---|---:|---:|---:|---:|
| SnowflakeMed Large 568M | `model_raw` | 0.38317654 | 0.45180136 | 0.40451025 | 0.21718282 |
| SnowflakeMed Large 568M | `framework_hybrid` | 0.71839115 | 0.79971517 | 0.74436588 | 0.68799290 |
| QwenMed XLarge 600M | `model_raw` | 0.65524910 | 0.55622778 | 0.62390935 | 0.42563684 |
| QwenMed XLarge 600M | `framework_hybrid` | 0.70703259 | 0.74758520 | 0.72000933 | 0.65551666 |
| BigMed Large 560M | `model_raw` | 0.36586781 | 0.43621053 | 0.38644388 | 0.20547663 |
| BigMed Large 560M | `framework_hybrid` | 0.75149926 | 0.80644766 | 0.76868518 | 0.67637119 |
| Privacy Filter Multilingual v2 | `model_raw` | 0.72285188 | 0.57795108 | 0.67750810 | 0.57674947 |
| Privacy Filter Multilingual v2 | `framework_hybrid` | 0.77419556 | 0.72623987 | 0.75843698 | 0.77277499 |

OpenMed 内部没有单一赢家：Privacy Filter 的 raw pooled micro/macro 最高；BigMed 的 hybrid pooled
micro 最高，而 Privacy Filter 的 hybrid pooled macro 最高。四个 raw pooled micro 都未超过已见
AIguard raw `0.7364`，四个 hybrid pooled micro 也都未超过已见 Presidio + CLUENER
`0.9474`。后两者来自不同 generation 的、已见公开合成 benchmark，只能作描述性参照，不是 active
SOTA gate，也不能据此进行结果驱动调参。PII Bench ZH 没有 PII-free 文档，因而这次 `8/8` 仍不能
给出部署 FPR；D1 与 human hidden 也均未完成，所以不能声明“首个”“最强”或 SOTA。公共聚合收据见
[`openmed_generation2_repair_v2_results.json`](../reports/baselines/openmed_generation2_repair_v2_results.json)。

Privacy Filter 的成功运行之前有三次 pre-prediction startup failure；现有启动日志至多表明后两次
可能已经通过 dataset-hash 检查，不能把它们计作结果执行。成功运行随后又由 launcher replay
复核：除 elapsed 字段外逐字段一致；launcher 源文件与 replay aggregate 文件 SHA-256 分别以
`e388a91f…` / `2bb0dc5d…` 开头。这些失败与 replay 都不改变上述 evidence classification 或 claim
禁用状态。

### 已封存的首轮迁移实验（只作诊断）

`aiguard_closed8_transfer_char_seed42_v1` 从固定 AIguard revision 迁移到本项目字符边界
tokenizer，并仅使用 16,000 条合成 train、2,000 条合成 validation 训练一轮。唯一一次
PII Bench ZH 运行发生在训练和 checkpoint 选择之后；从这次运行起，该 benchmark 对此实验族
记为 `test_already_seen`，禁止再用于阈值、checkpoint、损失或数据配方选择。

| 实验 | Formal micro/macro | Chat micro/macro | Pooled micro/macro | 门槛结论 |
|---|---:|---:|---:|---|
| AIguard -> project char-boundary pilot | 0.7726 / 0.7862 | 0.6756 / 0.5860 | 0.7426 / 0.7370 | pooled 超过两个公开模型，但 Formal micro 低于 AIguard 0.0288、Chat micro 低于 PII Engineer 0.0317；all-suite gate 失败 |

该行是 `experimental_pilot / diagnostic_only / not_release_candidate`，不能替代上表的独立公开
基线，也不能进入 release headline。聚合报告及其证据绑定位于
[`reports/experiments/aiguard_closed8_transfer_char_seed42_v1_pii_bench_zh.json`](../reports/experiments/aiguard_closed8_transfer_char_seed42_v1_pii_bench_zh.json)，
SHA-256 为 `65166c3c00e5f97acbce35151cc1c5747eca11b435a4ed146bc20671ba02cf13`。
聚合错误桶显示主要短板是 `ADDRESS` exact span（Formal F1 0.2390、Chat F1 0.0078），其次是
`PERSON_NAME`、`PHONE_NUMBER` 和车牌召回。后续只能针对 train/validation 的地址边界、聊天语境
和 hard negatives 优化；不得回看测试文本或按上述分数选择新配方。

报告分别位于 `reports/baselines/project_v1_closed8_pii_bench_zh.json`、
`reports/baselines/project_no_class_closed8_pii_bench_zh.json`、
`reports/baselines/aiguard_pii_bench_zh.json`、
`reports/baselines/openai_privacy_filter_pii_bench_zh.json`、
`reports/baselines/pii_engineer_chinese_ner_pii_bench_zh.json`、
`reports/baselines/presidio_rules_closed8_pii_bench_zh.json`、
`reports/baselines/argus_redact_pii_bench_zh.json`、
`reports/baselines/presidio_ernie_pii_bench_zh.json` 和
`reports/baselines/presidio_cluener_pii_bench_zh.json`。项目 no-class-weight 行是已有 JPT
checkpoint 的诊断，不是可发布 full-attention 候选，也没有参与 test 调参。

PII Engineer 的固定 artifact 实测为 `BertForTokenClassification` 的 BIO ONNX，而不是模型卡
文字所称的 GLiNER2。它的 frontmatter 声明 Apache-2.0、正文又声明 AGPL-3.0；本项目仅报告
固定 hash 上的独立结果，标记 `conflicting_upstream_declarations`，并明确
`redistribute_weights=false`。这类 artifact/文档冲突也是不能直接抄模型卡自报分数的原因。

AIguard 的 [聚合报告](../reports/baselines/aiguard_pii_bench_zh.json) 已在固定 revision、
解码前 closed-8 mask 和同一 strict evaluator 下复现：formal/chat/pooled strict micro F1 分别为
`0.80142013`、`0.57135430`、`0.73639614`。Formal/chat macro F1 为 `0.77363919` / `0.43105550`。
`ADDRESS` 在两套数据上的 exact F1 都为 0，说明边界错误不能用 relaxed overlap 掩盖。两套
PII Bench 均没有 PII-free 文档，因此这组结果不提供文档 FPR；也不得外推为 Open-24 或真实语境
质量。

OpenAI Privacy Filter 固定在 `7ffa9a043d54d1be65afb281eddf0ffbe629385b`，使用官方
`openai/privacy-filter` 源码 `f7f00ca7fb869683eb732c010299d901457f19c3` 的 constrained
BIOES Viterbi decoder 和 checkpoint 自带的 `default` 全零 operating point；没有调用普通
pipeline argmax，也没有在 PII Bench 上调阈值。预注册映射只覆盖银行卡、地址、邮箱、姓名和
电话，身份证、护照、车牌保持 unsupported 并计 FN；`private_date/private_url/secret` 的原始
logits 在 softmax 与 Viterbi 前屏蔽。Formal/Chat/Pooled strict micro F1 为
`0.58400567 / 0.40046106 / 0.53063277`，macro F1 为
`0.42899556 / 0.27801345 / 0.39049225`。该英语主模型在统一简中协议上没有改变 model-only
包络。聚合报告 SHA-256 为
`0461126c4f4b94b2657337325f72ba39f1b341fdc23de19d980947f1dbf5777f`；其中固定了七个模型文件、
官方 decoder、Transformers 实现、下载镜像与每个 artifact 的 hash。该执行仅是已见 benchmark
上的固定基线测量，禁止用于项目模型或阈值选择。

ERNIE 的模型单独轨 pooled F1 为 `0.56308368`；历史 custom-fusion hybrid pooled F1 为
`0.89599034`，而 release-native Presidio 2.2.363 v2 为 `0.89680396`。这说明本项目不能只与
裸 NER 比较后声称工程效果最佳，必须同时超过最强 model-only 与最强 framework-hybrid 两道门。
由于 PII Engineer 在 Chat 上达到 `0.70730994`，而 AIguard 在 Formal/Pooled 更强，
“model-only 最佳”必须超过逐 suite 的基线包络，不能只超过某一个模型的 pooled 值。
`cn_common_v5` 自身 pooled F1 为 `0.67099691`，precision 为 `0.99855221`、recall 为
`0.50525726`，解释了 Presidio hybrid 的主要互补来源；完整 rules-only 门槛已由下述 Argus
结果提高到 `0.92748927`。

CLUENER BIOS checkpoint 的 model-only pooled micro/macro F1 为
`0.57469388 / 0.24229805`，只覆盖 `ADDRESS` 与 `PERSON_NAME`。历史 custom-fusion 与
release-native v2 的 Formal/Chat/Pooled micro F1 在八位小数上均为
`0.95273279 / 0.93613793 / 0.94736372`；相对历史 ERNIE custom-fusion 的 pooled 增量是
`0.05137338`，相对 native ERNIE v2 的增量是 `0.05055976`。其 pooled
`ADDRESS`/`PERSON_NAME` F1 为 `0.98850575 / 0.94987864`，而银行卡 F1 仅
`0.17623498`，清楚显示互补来源与剩余短板。固定模型卡没有许可证声明，上游数据源还涉及
THUCTC/Sina News RSS，因此本项目只做固定 hash 推理比较并设置 `redistribute_weights=false`，
不能把该权重并入或派生为公开模型。

`argus-redact==0.7.18` fast mode 不加载 NER 或 LLM，以固定 closed-8 类型过滤在全量
5,000 Formal + 3,000 Chat 上得到 micro F1 `0.93549848 / 0.91071671`，pooled 为
`0.92748927`（macro `0.93205569`）。它把 rules-only 门槛从 `0.67099691` 大幅提高，但不超过
Presidio + CLUENER hybrid。该结果的解释边界更严格：framework 与 PII Bench ZH 同作者，固定
framework 仓库还含 benchmark 生成器和针对该数据的回归测试，因此只能称
`development-coupled synthetic coverage check`。聚合报告明确记录 coupling，且不含原文、实体值
或本地路径；SHA-256 为 `2cd12c69aec08544f99aa6e9af21dbd0bfcfc8701e7620d92213021d24e262bd`。

对 CLUENER hybrid 与 ERNIE hybrid 的同文档配对 bootstrap（1,000 次）给出
Formal/Chat/Pooled delta 95% 区间 `[0.0453, 0.0516]`、`[0.0518, 0.0634]`、
`[0.0484, 0.0541]`，所有 replicate 都是 CLUENER 胜出。这是结果已知后的
`post_hoc_descriptive_test_already_seen` 不确定性描述，不是预注册 confirmatory p-value，也不得
用于后续选择。项目 v1 full-system 因轨道不同、输出含 closed-8 外标签且缺少解码前/最终 mask
证据，被标为 `excluded_incompatible_prediction_contract`，没有通过事后过滤强行配对。聚合报告见
[`presidio_cluener_vs_ernie_paired_bootstrap.json`](../reports/baselines/presidio_cluener_vs_ernie_paired_bootstrap.json)。

Presidio 官方允许把 Hugging Face NER 作为 `NlpEngine` 或额外 `EntityRecognizer`；
但上述 ERNIE/CLUENER 历史数字实际来自 `LocalRecognizer` + custom fusion，不是原生
`AnalyzerEngine` 执行。因此它们只能表述为已声明的自定义框架配置，不是 Presidio
默认中文能力或 stock Presidio 结果。release-native v2 已独立重现 AnalyzerEngine 路径并冻结
result closure，关闭的是技术语义复现缺口；由于该证据 post-freeze、public-test-exposed、无
PII-free 文档且权重许可未解决，它仍不能改写旧 registry 或激活 release/claims。参考：
[Presidio Analyzer](https://microsoft.github.io/presidio/analyzer/) 与
[no-code recognizer 配置](https://microsoft.github.io/presidio/tutorial/08_no_code/)。

## 4. “最佳”主张的通过条件

对外至多只能在未来 successor active registry、hidden gold 解封前预注册且下述统计门
全部通过后，写“在该限定 registry 中最佳”，不能写无边界的“全社区/所有开源模型
最佳”。当前 Generation-2 blocked candidate freeze 不能作为 claim 依据。三个声明分别通过，
禁止混排：

1. `best model_raw`：checkpoint + 固定原生 decoder；不做 FPR 匹配或 test 阈值调整。在 v2
   hidden test 的每个预注册 suite 上，strict micro 与 macro F1 都超过 registry 中 model-raw
   包络，同时满足事前固定的绝对 PII-free FPR guardrail。
2. `best model_calibrated at matched FPR`：当前禁用，因为 registry 尚无在共同 validation 与共同
   FPR 协议下运行的外部 `model_calibrated` comparator。只有至少一个这样的外部对照完成并在
   hidden gold 访问前预注册后才能启用；届时阈值/温度只能在同一 validation 上拟合，hidden test
   的 FPR 匹配容差固定为绝对 `0.5` 个百分点，且不能写成 model-raw。
3. `best end-to-end system`：项目 full-system 必须逐 suite 同时超过 rules-only、
   framework-hybrid 和已登记外部 full-system 的 micro/macro 包络。撤回 generation-1 中，
   描述性 micro 包络来自 Presidio + CLUENER，macro 包络来自 development-coupled Argus；它们
   不是活跃 claim gate。release-native CLUENER v2 虽复现相同 micro 包络，仍是 post-freeze、
   public-test-exposed、non-activating 证据。当前 Generation-2 blocked candidate freeze 不激活
   重算；必须先关闭四项 blocker 并生成 successor active registry。
4. 三种声明都采用 intersection-union gate：对每个 admitted comparator、每个 required reporting
   scope，strict micro/macro F1 的 `candidate - comparator` 95% CI 下界必须严格大于 0；PII-free
   FPR 则要求 `comparator + 0.005 - candidate` 的 95% CI 下界不小于 0，并要求候选 FPR 点估计
   不超过 5%。F1 零差值不算 superiority；FPR 的 margin-adjusted 零边界只算 non-inferior。
5. FPR 分母只接受 successor registry 绑定的显式 eligibility manifest；协议 gold 为空、OOD-only
   或映射后为空均不能自动取得 PII-free 资格。atomic suites 必须由与 outcome 无关的预注册分区
   生成；reporting slices 可以重叠，但必须共享同一个 paired resample plan。
6. 三个训练 seed 全部公开；seed/checkpoint 选择规则在 hidden test 前按 validation 冻结，禁止按
   test 选择最好 seed，并要求全部 seed 的方向性结论一致。
7. Closed-8 外部集、Open-24 hidden test、至少一个人类自然语境 hard-negative 集方向一致；
   标签覆盖、许可证缺失、训练集可能重叠、默认绑定/部署条件全部披露。

若任一门未通过，回到数据与模型迭代：优先补 human-context hard negatives 和低召回标签，随后
比较 class weight、BIOES+CRF、span head、窗口策略与 validation-only calibration。不得通过删除
弱标签、在 test 上调阈值或只展示 full-system 来制造领先。
