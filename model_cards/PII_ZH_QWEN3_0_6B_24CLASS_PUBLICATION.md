---
language:
- zh
library_name: transformers
pipeline_tag: token-classification
license: apache-2.0
base_model: ZJUICSR/AIguard-pii-detection-fast
base_model_revision: 677a5ebc1600fef61e8973cafd3026be322b3a73
tags:
- pii
- chinese
- token-classification
- qwen3
- full-attention
- bie
model_name: pii-zh-qwen3-0.6b-24class-bie73
package_version: 0.3.0rc1
---

# pii-zh-qwen3-0.6b-24class BIE73

面向简体中文的 24 类 PII 字符跨度检测模型。模型采用约 0.6B 参数的 Qwen3 token
classifier，将注意力改为 full attention，并通过 `O + 24 × B/I/E = 73` 标签与
constrained-Viterbi 解码生成合法、可复现的字符跨度。

模型可单独使用，也可通过
[`whyiug/pii-detect-model`](https://github.com/whyiug/pii-detect-model) 接入中文 PII 服务。
公开服务以 native Presidio 中文识别链为主检测器，BIE73 仅补充主检测器尚未覆盖的实体类型，
不会替换 Presidio 已负责类型的结果。

## 快速开始

推荐使用项目提供的 BIE73 加载器。它会校验模型结构、标签合同和本地权重，并固定使用
constrained-Viterbi。

```bash
python -m pip install \
  "pii-zh-qwen[inference] @ https://github.com/whyiug/pii-detect-model/releases/download/v0.3.0rc1/pii_zh_qwen-0.3.0rc1-py3-none-any.whl"
python -m pip install "huggingface_hub>=0.27"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MODEL="$PWD/models/pii-zh-qwen3-0.6b-24class"
hf download Forrest20231206/pii-zh-qwen3-0.6b-24class \
  --revision v0.3.0rc1 \
  --local-dir "$MODEL"
```

```python
from pii_zh.full_bie73 import load_full_bie73_predictor

predictor = load_full_bie73_predictor(
    "models/pii-zh-qwen3-0.6b-24class",
    scope="open24",
    device="cpu",
)
text = "联系人张三，邮箱 demo@example.com，电话 13800138000。"
print(predictor.predict(text))
```

每个结果包含 `entity_type`、左闭右开的字符 `start`/`end`、`score` 和解码元数据。批量场景可用
`predictor.predict_batch(texts)`。

> 不要使用 `transformers.pipeline(..., aggregation_strategy="simple")` 的通用 token 聚合来替代
> 上述加载器。通用聚合不执行模型所需的 constrained-Viterbi 状态约束，也不保证与 Open-24 /
> Closed-8 解码前 scope 一致，因此可能改变 span 边界或生成非法 B/I/E 路径。直接使用
> `AutoModelForTokenClassification` 只适合获取 raw logits，自行实现完整解码协议。

## 模型规格

| 项目 | 规格 |
|---|---|
| 参数量 | 616,384,658（约 0.6B） |
| 架构 | `Qwen3BiForTokenClassification`，28 层，hidden size 1024 |
| 注意力 | padding-aware full attention，SDPA，不使用 causal mask |
| 标签 | `O + 24 × B/I/E = 73` |
| 解码 | constrained-Viterbi；只允许合法 `O/B/I/E` 路径 |
| 最大训练长度 | 512 tokens |
| 发布权重 | 合并后的单一 safetensors，无运行时 LoRA adapter 依赖 |
| 初始化模型 | `ZJUICSR/AIguard-pii-detection-fast` |

full attention 让实体 token 同时使用左右文。例如值右侧出现的“手机号”“学号”“病历号”等提示
可直接参与当前 span 判断；BIE 约束负责稳定多 token 实体边界。

## 24 类标签

| 类别 | 实体类型 |
|---|---|
| 基本信息与联系方式 | `PERSON_NAME`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, `ADDRESS`, `DATE_OF_BIRTH` |
| 身份证件 | `CN_RESIDENT_ID`, `PASSPORT_NUMBER`, `DRIVER_LICENSE_NUMBER`, `SOCIAL_SECURITY_NUMBER` |
| 金融账户 | `BANK_CARD_NUMBER`, `BANK_ACCOUNT_NUMBER`, `ALIPAY_ACCOUNT` |
| 车辆、组织与医疗 | `VEHICLE_LICENSE_PLATE`, `EMPLOYEE_ID`, `STUDENT_ID`, `MEDICAL_RECORD_NUMBER` |
| 社交账号、网络与设备 | `WECHAT_ID`, `QQ_NUMBER`, `USERNAME`, `IP_ADDRESS`, `MAC_ADDRESS`, `DEVICE_ID` |
| 位置与安全信息 | `GEO_COORDINATE`, `SECRET` |

`SECRET` 表示密码、token、密钥等敏感字符串，不一定属于法律定义中的个人信息。

## 训练数据

本轮数据共 88,000 条，全部由项目的 clean-room 合成器生成。

| split | 文档数 | 进入梯度训练 | 用途 |
|---|---:|---:|---|
| train | 72,000 | 72,000 | 模型训练 |
| dev | 8,000 | 0 | checkpoint、seed 与服务策略选择 |
| independent test | 8,000 | 0 | 模型冻结后的一次性测试 |

训练、开发和独立测试均包含 60% 正样本与 40% PII-free hard negatives；24 类在每个 split 内
均衡。文本覆盖 12 个应用域、自然对话、叙述、流程、多实体和半结构化表达，并包含上下文明确为
非个人用途的 counterfactual hard negatives。

三个 split 在语法组、模板族、语义骨架、实体值组、来源组、文本哈希和文档 ID 上两两零碰撞。
本轮微调没有加入真实 PII、客户/生产数据、人工语料、公开 NER 训练行或 PII Bench 行。

“本轮微调数据全部合成”不代表模型全部知识来自合成数据。模型初始化自
`ZJUICSR/AIguard-pii-detection-fast` revision
`677a5ebc1600fef61e8973cafd3026be322b3a73`，并继承该模型及其 Qwen3 backbone 的上游权重。

## 训练方式

- seeds：`13 / 42 / 97`，每个 seed 训练 2 epochs；
- LoRA rank 32、alpha 64、dropout 0.05，覆盖 attention 与 MLP projection；分类头同时训练；
- 可训练参数 20,259,913，约占总参数 3.29%；
- effective train batch 64，AdamW + linear scheduler，warmup ratio 0.05；
- LoRA learning rate `1e-5`，分类头 learning rate `5e-5`，bf16 autocast；
- 统一开发集协议选择 seed 42、epoch 1、global step 1125，随后合并权重。

完整数据生成、隔离、选择规则与实验限制见 GitHub 的
[模型数据与训练技术报告](https://github.com/whyiug/pii-detect-model/blob/v0.3.0rc1/docs/model-training-technical-report.md)。

## 单模型评测

指标使用字符级 exact start/end/label 匹配。micro F1 汇总全部 span，macro F1 对类别等权平均；
PII-free document FPR 表示无 gold PII 的文档中至少出现一个预测 span 的比例。

| 数据集 | 文档数 | strict micro F1 | strict macro F1 | PII-free 文档 FPR |
|---|---:|---:|---:|---:|
| v7.2 dev synthetic | 8,000 | 0.956001 | 0.960702 | 0.067500 |
| v7.2 independent synthetic | 8,000 | 0.952047 | 0.956372 | 0.031250 |
| synthetic-v1.3 test | 2,000 | 0.739176 | 0.795211 | 0.253333 |

v7.2 independent 的 Open-24 指标接近开发集，但 Closed-8 relevant micro F1 相对开发集下降
`0.030529`，略超过预设的 `0.03` 稳定性范围。因此不能把表中的 Open-24 结果描述为“全部独立
门槛已通过”。不同合成器上的 FPR 差异也说明真实部署前必须进行目标域验证。

### PII Bench ZH

下表来自同批次冻结单模型报告。PII Bench ZH 是公开、程序化生成的 closed-8 benchmark，且不含
PII-free 文档；这些数字不能代表完整 Open-24、真实业务误报率或全局排名。

| split | strict micro F1 | strict macro F1 |
|---|---:|---:|
| Formal | `0.5465838509` | `0.5930315483` |
| Chat | `0.4749034749` | `0.4826017924` |
| Pooled | `0.5234460196` | `0.5660017265` |

同一数据 revision、closed-8 标签映射和 exact-span 协议下，部分开源模型的 independently
reproduced strict micro F1 如下。各模型使用其原生解码或固定适配器，没有用本 benchmark 调阈值。

| 开源模型 | Formal | Chat | Pooled |
|---|---:|---:|---:|
| `ZJUICSR/AIguard-pii-detection-fast` | `0.8014` | `0.5714` | `0.7364` |
| `OpenMed/privacy-filter-multilingual-v2` | `0.7229` | `0.5780` | `0.6775` |
| `OpenMed/OpenMed-PII-Chinese-QwenMed-XLarge-600M-v1` | `0.6552` | `0.5562` | `0.6239` |
| 本模型 BIE73 | `0.5466` | `0.4749` | `0.5234` |

因此本模型不作“最佳单模型”或 SOTA 声明。它在产品中的定位是扩展 Presidio 主检测器的类型覆盖，
而不是替换已经成熟的 closed-8 检测链。

## Presidio-primary 服务（不归因于模型权重）

GitHub 项目的 `community-presidio-bie73-cascade-v1` 采用 Presidio-primary + BIE73
augmentation。native Presidio 负责 closed-8；BIE73 在解码前只允许另外 16 个 Open-24 标签，
同类冲突不由模型改写主路由结果。统一服务再完成字符偏移规范化、validator 与确定性 fusion。

服务 builder 同时需要 BIE73 与固定 CLUENER 的本地目录：

```bash
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PRIMARY_SOURCE="models/uer/cluener2020-upstream"
PRIMARY_MODEL="models/uer/cluener2020-primary"
hf download uer/roberta-base-finetuned-cluener2020-chinese \
  README.md config.json pytorch_model.bin special_tokens_map.json \
  tokenizer_config.json vocab.txt \
  --revision cddd8fc233e373855a8c0a7f4b7eb83acb686a2b \
  --local-dir "$PRIMARY_SOURCE"
pii-zh prepare-cluener-primary \
  --source-dir "$PRIMARY_SOURCE" \
  --output-dir "$PRIMARY_MODEL"
```

准备命令会校验固定 revision，把 CLUENER 的 legacy 权重转换为服务使用的 safetensors-only 目录；
原始下载目录保持不变。

```python
from pii_zh.full_bie73 import build_full_bie73_service_pipeline

pipeline = build_full_bie73_service_pipeline(
    model_path="models/pii-zh-qwen3-0.6b-24class",
    primary_model_path="models/uer/cluener2020-primary",
    device="cpu",
)
```

冻结 native Presidio closed-8 基线如下。这是主检测器分支本身的分数，不是 BIE73 权重分数。

| split | strict micro F1 | strict macro F1 |
|---|---:|---:|
| Formal | `0.9527327902` | `0.8732108951` |
| Chat | `0.9361379310` | `0.8399701577` |
| Pooled | `0.9473637236` | `0.8667338443` |

实现合同会在 BIE73 解码前屏蔽全部 closed-8 标签，原样保留 primary 输出，并让 primary 赢得跨
分支 overlap；因此组合服务的 closed-8 投影与该 primary 分支完全相同。上述 `0.9473637236`
不能称为整套 Open-24 服务 F1：当前 Open-24 增强服务尚无新的严格整体 F1。这组证据不支持
“增强服务严格超越”或全局 SOTA 声明，补充类型仍应在目标域单独验证。

Presidio primary 依赖固定 revision 的 `uer/roberta-base-finetuned-cluener2020-chinese`。该权重
不随本模型或 Python 包分发，用户需自行下载到本地；其上游模型卡未声明模型许可证，使用或再分发
前请先审阅上游材料与适用条款。

## 使用范围与限制

- 适用于简体中文文本中的候选 PII span 检测，不提供法律或合规判断；
- 训练数据全部合成，无法完整覆盖真实书写习惯、行业缩写、OCR/ASR 噪声和分布漂移；
- 模型可能漏检或误检；高风险脱敏流程应使用 Presidio-primary 集成服务、目标域评测和必要的
  人工复核；
- 单模型不校验号码是否真实签发或属于某人；validator 也只证明格式相容；
- 长文档应使用项目级联服务提供的窗口化推理，不建议直接截断；
- 应避免在应用日志中记录输入原文或命中的敏感值。

## 许可证与来源

- 项目代码：Apache License 2.0；
- 初始化模型：`ZJUICSR/AIguard-pii-detection-fast`，具体条款以其模型仓库为准；
- 完整第三方来源与再分发说明：GitHub 仓库中的 `NOTICE` 与 `THIRD_PARTY_NOTICES.md`。

项目仓库：https://github.com/whyiug/pii-detect-model
