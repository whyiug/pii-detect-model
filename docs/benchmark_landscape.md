# 中文 PII 检测公开比较

本页汇总本项目在统一协议下复现的中文 PII 模型、规则和框架结果。模型与完整服务分轨展示：
模型分数只代表可下载权重；框架分数包含 NER、规则、validator 与框架自身的重叠处理，不能归因
给单个模型。

## 评测协议

主表使用 [PII Bench ZH](https://huggingface.co/datasets/wan9yu/pii-bench-zh) 固定 revision
`c350b94897af668517ff5de237d89f2ce2eaa6f0`：

- Formal 5,000 篇，Chat 3,000 篇；
- 共同的 closed-8 标签：地址、银行卡、身份证、邮箱、护照、姓名、电话和车牌；
- 字符级 exact `(start, end, label)` 匹配；
- 表中数值均为 strict micro F1；
- 不在共同标签内的模型输出在解码前屏蔽，未支持的共同标签按漏检计入；
- 每个对象使用固定 revision、原生解码或项目内固定适配器，不用该 benchmark 调阈值。

PII Bench ZH 是公开的程序化合成 benchmark，且全部文档都含协议内 PII，因此它适合做可复现的
同协议比较，但不能测量 PII-free 文档误报率，也不能替代真实业务或隐藏测试。

## 开源模型对比

| 模型 | 共同标签覆盖 | Formal | Chat | Pooled |
|---|---:|---:|---:|---:|
| [ZJUICSR/AIguard-pii-detection-fast](https://huggingface.co/ZJUICSR/AIguard-pii-detection-fast) | 8/8 | **0.8014** | 0.5714 | **0.7364** |
| PII Engineer Chinese NER | 4/8 | 0.7183 | **0.7073** | 0.7148 |
| [OpenMed Privacy Filter Multilingual v2](https://huggingface.co/OpenMed/privacy-filter-multilingual-v2) | 8/8 | 0.7229 | 0.5780 | 0.6775 |
| [OpenMed QwenMed XLarge 600M](https://huggingface.co/OpenMed/OpenMed-PII-Chinese-QwenMed-XLarge-600M-v1) | 8/8 | 0.6552 | 0.5562 | 0.6239 |
| CLUENER2020 RoBERTa | 2/8 | 0.5753 | 0.5735 | 0.5747 |
| OpenAI Privacy Filter | 5/8 | 0.5840 | 0.4005 | 0.5306 |
| **本项目 BIE73 0.6B** | 8/8 | 0.5466 | 0.4749 | 0.5234 |

BIE73 在该 closed-8 benchmark 上不是最佳单模型。它的主要价值是提供 24 类简体中文标签、
full-attention BIE73 解码和可审计的本地推理接口，并在产品服务中补充 primary 未覆盖的 16 类，
而不是替换成熟的 closed-8 检测链。

BIE73 的完整 strict micro/macro F1 为：

| Split | Micro F1 | Macro F1 |
|---|---:|---:|
| Formal | 0.5465838509 | 0.5930315483 |
| Chat | 0.4749034749 | 0.4826017924 |
| Pooled | 0.5234460196 | 0.5660017265 |

## 开源规则与框架对比

下表是独立的 framework/rules 赛道，仍使用相同 closed-8 gold 和 exact-span 评分。

| 系统 | 类型 | Formal | Chat | Pooled |
|---|---|---:|---:|---:|
| **Presidio 2.2.363 + CLUENER + `cn_common_v5`** | framework hybrid | **0.9527** | **0.9361** | **0.9474** |
| argus-redact 0.7.18 fast | rules/framework | 0.9355 | 0.9107 | 0.9275 |
| Presidio 2.2.363 + ERNIE + `cn_common_v5` | framework hybrid | 0.9052 | 0.8793 | 0.8968 |
| OpenMed Privacy Filter Multilingual hybrid | framework hybrid | 0.7742 | 0.7262 | 0.7584 |
| OpenMed QwenMed hybrid | framework hybrid | 0.7070 | 0.7476 | 0.7200 |
| `cn_common_v5` | rules only | 0.6780 | 0.6563 | 0.6710 |

这组结果说明，在 PII Bench ZH 的常见八类上，规则与 NER 的组合明显优于当前 BIE73 单模型。
因此公开服务采用第一行作为 primary，并让 BIE73 只负责另外 16 类 augmentation。

冻结 primary 的完整结果为：

| Split | Micro F1 | Macro F1 |
|---|---:|---:|
| Formal | 0.9527327902 | 0.8732108951 |
| Chat | 0.9361379310 | 0.8399701577 |
| Pooled | 0.9473637236 | 0.8667338443 |

`community-presidio-bie73-cascade-v1` 在模型解码前屏蔽全部 closed-8 标签，原样保留 primary
输出，并让 primary 赢得跨分支重叠。因此组合服务的 closed-8 投影与该冻结 primary 相同。
`0.9473637236` 仍是 closed-8 primary 证据，不是整套 Open-24 服务 F1；augmentation-16
目前没有新的统一 Open-24 严格整体分数。

## 跨数据集回归

| 数据集 | 结果 | 说明 |
|---|---:|---|
| CLUENER | strict micro F1 0.794555（5020/6318） | 超过第一版 0.786972；BIE73 不参与该兼容分支 |
| CrossWOZ | 2 emitting docs / 2 spans | 对照为 8 / 8；无 span gold，不报告 F1 |

CLUENER 用于确认既有中文 NER/集成能力没有退化。CrossWOZ 仅提供输出数量回归，不能据此推断召回
或整体准确率。

## 如何解读这些结果

- 单模型、规则和 framework hybrid 必须分轨比较。
- 公开合成 benchmark 的分数不等同于真实业务效果。
- 标签覆盖不同会直接影响 F1；表中“覆盖”用于解释可比范围，不用于豁免漏检。
- BIE73 的 88,000 条本轮微调数据全部合成；模型仍继承 AIguard 与 Qwen3 backbone 的上游知识。
- 正式部署前应使用目标域人工复核数据评估误报、漏报、延迟和 PII-free 文档 FPR。

详细的数据生成与训练方式见
[模型数据与训练技术报告](model-training-technical-report.md)，默认服务的安装与 API 见
[Presidio-primary 集成服务](community-model-service.md)。

## 可复现证据

- [BIE73 模型卡](../model_cards/PII_ZH_QWEN3_0_6B_24CLASS_PUBLICATION.md)
- [结构化 model-index](../model_cards/model-index.yml)
- [AIguard 聚合报告](../reports/baselines/aiguard_pii_bench_zh.json)
- [OpenMed 聚合报告](../reports/baselines/openmed_generation2_repair_v2_results.json)
- [native Presidio + 中文 NER 聚合报告](../reports/baselines/native_presidio_zh_ner_framework_hybrid_v2.json)

第三方模型作者自报的指标可能使用不同数据、标签和评分方式；除非明确说明，本页只展示本项目在
上述统一协议下独立复现的结果。
