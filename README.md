# pii-zh-qwen

面向简体中文的本地 PII 检测模型与集成服务。服务以 Presidio 中文识别链为主检测器，保留其在
常见身份、联系方式和证件类型上的稳定输出；约 0.6B 参数的 full-attention BIE73 模型只补充
主检测器尚未覆盖的类型。CLI、Python API 与 FastAPI 使用统一的字符级 span 输出。

[![Release](https://img.shields.io/github/v/release/whyiug/pii-detect-model?include_prereleases)](https://github.com/whyiug/pii-detect-model/releases/tag/v0.3.0rc1)
[![Model](https://img.shields.io/badge/Hugging%20Face-model-FFD21E)](https://huggingface.co/Forrest20231206/pii-zh-qwen3-0.6b-24class)
[![License](https://img.shields.io/github/license/whyiug/pii-detect-model)](LICENSE)

## 主要能力

- 24 类中文 PII：姓名、联系方式、证件、金融账号、组织编号、网络与设备标识等；
- full attention + `O + 24 × B/I/E = 73` 标签，使用 constrained-Viterbi 输出合法字符跨度；
- Presidio-primary + BIE73 augmentation：模型不覆盖或改写 Presidio 已负责的类型；
- 开箱即用的 `community-presidio-bie73-cascade-v1` 集成 profile；
- 检测、脱敏、批量 JSONL、Python API、Presidio 接入和本地 HTTP 服务；
- 模型只从显式指定的本地目录加载，CLI 与服务不会自动上传文本或下载权重。

## 快速开始

需要 Python 3.10 或更高版本。

```bash
python -m pip install \
  "pii-zh-qwen[cascade,service] @ https://github.com/whyiug/pii-detect-model/releases/download/v0.3.0rc1/pii_zh_qwen-0.3.0rc1-py3-none-any.whl"
python -m pip install "huggingface_hub>=0.27"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
BIE73_MODEL="$PWD/models/pii-zh-qwen3-0.6b-24class"
PRIMARY_SOURCE="$PWD/models/uer/cluener2020-upstream"
PRIMARY_MODEL="$PWD/models/uer/cluener2020-primary"

hf download Forrest20231206/pii-zh-qwen3-0.6b-24class \
  --revision v0.3.0rc1 \
  --local-dir "$BIE73_MODEL"
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

### Python

默认构建 Presidio-primary 服务。BIE73 使用 constrained-Viterbi，只在主检测器未覆盖的实体类型上
提供 augmentation；同类冲突不由模型推翻 Presidio 结果。

```python
from pii_zh.full_bie73 import build_full_bie73_service_pipeline

pipeline = build_full_bie73_service_pipeline(
    model_path="models/pii-zh-qwen3-0.6b-24class",
    primary_model_path="models/uer/cluener2020-primary",
    device="cpu",
)
text = "联系人张三，手机号 13800138000，邮箱 demo@example.com。"
print([span.to_dict() for span in pipeline.detect(text)])
print(pipeline.redact(text, "<PII>"))
```

### CLI

`--mode`、`--scope` 和阈值均可省略；该 profile 默认启用 Presidio 主检测与 BIE73 补充路径。

```bash
pii-zh detect \
  --profile community-presidio-bie73-cascade-v1 \
  --model-path "$BIE73_MODEL" \
  --primary-model-path "$PRIMARY_MODEL" \
  --text '联系人张三，手机号 13800138000。' \
  --pretty

pii-zh redact \
  --profile community-presidio-bie73-cascade-v1 \
  --model-path "$BIE73_MODEL" \
  --primary-model-path "$PRIMARY_MODEL" \
  --text '联系人张三，手机号 13800138000。' \
  --replacement '<PII>'
```

有 CUDA 时可显式传入 `--device cuda`。项目不会替用户选择、抢占或清理 GPU。

### HTTP

```bash
pii-zh serve \
  --profile community-presidio-bie73-cascade-v1 \
  --model-path "$BIE73_MODEL" \
  --primary-model-path "$PRIMARY_MODEL" \
  --host 127.0.0.1 \
  --port 8000
```

```bash
curl --fail --silent --show-error \
  -H 'content-type: application/json' \
  --data '{"text":"测试邮箱 demo@example.com"}' \
  http://127.0.0.1:8000/v1/analyze
```

另有 `/healthz` 与 `/v1/redact`。服务默认仅监听 loopback，限制文本长度、请求体、并发和超时，
并关闭 access log。

## 为什么不能直接使用通用 token-classification pipeline

BIE73 的最终 span 依赖合法状态转移和解码前标签 scope。请使用项目提供的
`load_full_bie73_predictor` 或级联 builder；不要把
`transformers.pipeline(..., aggregation_strategy="simple")` 的通用聚合结果作为本模型输出。
通用聚合不执行项目的 constrained-Viterbi 约束，可能改变边界并产生非法 B/I/E 路径。

## 支持的实体类型

| 类别 | 实体类型 |
|---|---|
| 基本信息与联系方式 | `PERSON_NAME`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, `ADDRESS`, `DATE_OF_BIRTH` |
| 身份证件 | `CN_RESIDENT_ID`, `PASSPORT_NUMBER`, `DRIVER_LICENSE_NUMBER`, `SOCIAL_SECURITY_NUMBER` |
| 金融账户 | `BANK_CARD_NUMBER`, `BANK_ACCOUNT_NUMBER`, `ALIPAY_ACCOUNT` |
| 车辆、组织与医疗 | `VEHICLE_LICENSE_PLATE`, `EMPLOYEE_ID`, `STUDENT_ID`, `MEDICAL_RECORD_NUMBER` |
| 社交账号、网络与设备 | `WECHAT_ID`, `QQ_NUMBER`, `USERNAME`, `IP_ADDRESS`, `MAC_ADDRESS`, `DEVICE_ID` |
| 位置与安全信息 | `GEO_COORDINATE`, `SECRET` |

`SECRET` 用于识别密码、token、密钥等敏感字符串，不一定属于法律定义中的个人信息。

## 效果概览

所有分数均使用字符级 exact-span 匹配。单模型与 Presidio 服务是两个独立轨道，不能把服务分数
归因于模型权重，也不能把某一轨道的结果外推到另一轨道。

| 评测 | 轨道 | strict micro F1 | strict macro F1 | PII-free 文档 FPR |
|---|---|---:|---:|---:|
| v7.2 independent synthetic | 单模型 Open-24 | 0.952047 | 0.956372 | 0.031250 |

跨数据集服务回归结果：

| 数据集 | 结果 | 说明 |
|---|---:|---|
| CLUENER | strict micro F1 `0.794555` | 历史兼容分支；B3 模型不参与该分支 |
| CrossWOZ | `2` emitting docs / `2` spans | 对照服务为 `8 / 8`；该集合无 span-level gold，不报告 F1 |

PII Bench ZH closed-8 结果如下。该公开 benchmark 不含 PII-free 文档，不能代表完整 Open-24、
真实业务误报率或全局排名。

| 轨道 | Formal micro / macro | Chat micro / macro | Pooled micro / macro |
|---|---|---|---|
| BIE73 单模型 | `0.5465838509 / 0.5930315483` | `0.4749034749 / 0.4826017924` | `0.5234460196 / 0.5660017265` |
| 冻结 native Presidio 主检测基线 | `0.9527327902 / 0.8732108951` | `0.9361379310 / 0.8399701577` | `0.9473637236 / 0.8667338443` |

公开服务以第二行的 native Presidio 方案为主路由，BIE73 只补充其未覆盖的 16 类。实现合同在
解码前屏蔽模型的 closed-8 标签，原样保留 primary 输出且跨分支重叠由 primary 获胜，因此组合
服务的 closed-8 投影与冻结 primary 分支相同。这里的数字仍是 primary 分支的冻结证据，不是整套
Open-24 增强服务的新评测；Open-24 尚无新的严格整体 F1。项目不作严格超越或 SOTA 声明。

同协议的 model-only 对照中，AIguard fast 的 pooled micro F1 为 `0.7364`，OpenMed QwenMed
XLarge 600M 为 `0.6239`，OpenMed Privacy Filter Multilingual 为 `0.6775`；均高于本模型的
`0.5234`。这也是默认产品采用“强 Presidio primary + BIE73 扩类”而不是替换主检测器的原因。
完整的开源模型、规则框架和 hybrid 分轨比较见
[公开比较说明](docs/benchmark_landscape.md)。

## 模型与训练数据

- 初始化自 `ZJUICSR/AIguard-pii-detection-fast` 的固定 revision，并继承其上游权重知识；
- 约 616M 参数，full-attention Qwen3 token classifier，73 个 BIE 标签；
- 本轮数据共 88,000 条，全部由项目生成：72,000 train、8,000 dev、8,000 independent test；
- 只有 72,000 条 train 进入梯度训练；未加入真实 PII、客户数据或公开 NER 训练行；
- 三个 split 在语法组、模板族、语义骨架、实体值组、来源组、文本哈希和文档 ID 上隔离。

Presidio primary 使用固定 revision 的
`uer/roberta-base-finetuned-cluener2020-chinese`。该权重不随 Python 包或 BIE73 模型包分发，
必须由使用者下载到本地；其上游模型卡未声明模型许可证，使用和再分发前请自行审阅上游材料与
适用条款。

全合成数据具有可审计和无真实 PII 的优点，但不能替代真实业务域、人工复核 hidden test 或分布
漂移评估。完整训练方式见[模型数据与训练技术报告](docs/model-training-technical-report.md)。

## 使用建议

- 一般应用优先使用 Presidio-primary 默认服务；BIE73 augmentation 应限制在主检测器未覆盖类型。
- `mode="model-only"` 是仍保留阈值、validator 和 fusion 的服务消融，不是 raw-model benchmark
  路径，也不是产品默认配置。
- 上线前用目标域数据验证标签定义、误报、漏报、延迟和资源占用，并保留人工抽检。
- `start`/`end` 是左闭右开的字符偏移；输出不回显命中的原文。
- 模型会误检和漏检，检测结果不等同于法律或合规结论。

## 文档

- [Hugging Face Model Card](https://huggingface.co/Forrest20231206/pii-zh-qwen3-0.6b-24class)
- [Presidio-primary 集成服务](docs/community-model-service.md)
- [模型数据与训练技术报告](docs/model-training-technical-report.md)
- [License](LICENSE) 与 [第三方许可](THIRD_PARTY_NOTICES.md)

## License

仓库代码采用 [Apache License 2.0](LICENSE)。模型初始化权重、数据与依赖的第三方条款独立适用，
再分发或生产使用前请阅读 [NOTICE](NOTICE) 与 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。
