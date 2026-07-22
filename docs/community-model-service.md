# Presidio-primary 中文 PII 服务与 BIE73 扩展

`community-presidio-bie73-cascade-v1` 面向需要稳定中文 PII 检测和更广类型覆盖的应用。公开服务采用
Presidio-primary + BIE73 augmentation：native Presidio 中文识别链负责已覆盖类型，BIE73 只补充
主检测器未覆盖的类型，最后统一字符偏移、validator 和 span fusion。

这种分工保留 Presidio 在常见身份、联系方式和证件类型上的成熟能力，同时让 BIE73 扩展组织编号、
医疗编号、账号、设备和其他上下文相关类型。模型不会覆盖或改写 Presidio 已负责类型的结果。

## 默认合同

| 配置 | 默认值 |
|---|---|
| profile | `community-presidio-bie73-cascade-v1` |
| primary detector | native Presidio 中文识别链 |
| augmentation | BIE73，仅允许主检测器未覆盖的实体类型 |
| model decoder | `constrained_viterbi` |
| model scope | Open-24 减 closed-8，共 16 类，解码前约束 |
| model threshold | 启用的补充类型固定为 `0.5` |
| overlap | Presidio 已覆盖类型优先；补充类型再做确定性 fusion |

PII Bench ZH 的 closed-8 标签为 `ADDRESS`、`BANK_CARD_NUMBER`、`CN_RESIDENT_ID`、
`EMAIL_ADDRESS`、`PASSPORT_NUMBER`、`PERSON_NAME`、`PHONE_NUMBER` 和
`VEHICLE_LICENSE_PLATE`。公开服务不会使用 BIE73 改写这些主路由结果。对其余 Open-24 标签，
只有未被 Presidio 配置覆盖的类型才进入模型 augmentation allow-list。

## 安装与模型下载

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

服务只从显式指定的本地目录加载模型。`prepare-cluener-primary` 校验固定 revision，将上游
`pytorch_model.bin` 转为经过验证的 safetensors-only 目录，且不修改下载目录。CLUENER 权重不随
Python 包或 BIE73 模型包分发；其上游模型卡未声明模型许可证，使用者需自行下载并在使用或再分发
前审阅上游材料与适用条款。

## Python API

```python
from pii_zh.full_bie73 import build_full_bie73_service_pipeline

pipeline = build_full_bie73_service_pipeline(
    model_path="models/pii-zh-qwen3-0.6b-24class",
    primary_model_path="models/uer/cluener2020-primary",
    device="cpu",
)

text = "联系人张三，手机号 13800138000，病历号 MR-2026-001。"
detections = [item.to_dict() for item in pipeline.detect(text)]
redacted = pipeline.redact(text, "<PII>")
```

检测结果不回显命中的原文，包含左闭右开的字符 `start`/`end`、`entity_type`、`score`、主要来源
`source`、参与来源 `sources` 与 `decision_process`。可通过 `entities` 只返回指定类型：

```python
pipeline.detect(text, entities=["PERSON_NAME", "MEDICAL_RECORD_NUMBER"])
```

批量调用使用 `pipeline.detect_batch(texts)`。

### 单模型推理

单独研究模型跨度时使用项目加载器：

```python
from pii_zh.full_bie73 import load_full_bie73_predictor

predictor = load_full_bie73_predictor(
    "models/pii-zh-qwen3-0.6b-24class",
    scope="open24",
    device="cpu",
)
spans = predictor.predict("病历号 MR-2026-001，设备号 DEV-A1024。")
```

不要使用 `transformers.pipeline(..., aggregation_strategy="simple")` 替代该加载器。BIE73 的最终
span 依赖 constrained-Viterbi 和解码前标签 scope，通用 token 聚合不保证合法 B/I/E 路径。

`mode="model-only"` 仍会保留服务阈值、validator 和 fusion，只跳过 primary 分支；它不是产品默认
配置，也不等同于 raw-model benchmark。

## CLI

```bash
pii-zh detect \
  --profile community-presidio-bie73-cascade-v1 \
  --model-path "$BIE73_MODEL" \
  --primary-model-path "$PRIMARY_MODEL" \
  --text '联系人张三，手机号 13800138000，病历号 MR-2026-001。' \
  --pretty

pii-zh redact \
  --profile community-presidio-bie73-cascade-v1 \
  --model-path "$BIE73_MODEL" \
  --primary-model-path "$PRIMARY_MODEL" \
  --text '联系人张三，手机号 13800138000，病历号 MR-2026-001。' \
  --replacement '<PII>'
```

批量 JSONL 输入每行只包含一个 `text` 字段：

```bash
pii-zh detect \
  --profile community-presidio-bie73-cascade-v1 \
  --model-path "$BIE73_MODEL" \
  --primary-model-path "$PRIMARY_MODEL" \
  --input-file input.jsonl \
  --jsonl > output.jsonl
```

有 CUDA 时显式增加 `--device cuda`。CLI 不会自行选择、占用或清理 GPU。

## HTTP API

```bash
pii-zh serve \
  --profile community-presidio-bie73-cascade-v1 \
  --model-path "$BIE73_MODEL" \
  --primary-model-path "$PRIMARY_MODEL" \
  --device cpu \
  --host 127.0.0.1 \
  --port 8000
```

```bash
curl --fail --silent --show-error \
  -H 'content-type: application/json' \
  --data '{"text":"测试邮箱 demo@example.com，病历号 MR-2026-001。"}' \
  http://127.0.0.1:8000/v1/analyze
```

脱敏接口：

```bash
curl --fail --silent --show-error \
  -H 'content-type: application/json' \
  --data '{"text":"测试邮箱 demo@example.com","replacement":"<PII>"}' \
  http://127.0.0.1:8000/v1/redact
```

另有 `GET /healthz`。服务默认只监听 loopback，固定单 worker，关闭 access log，并限制请求体、文本
长度、并发和超时。CPU 默认并发为 4，CUDA 默认并发为 1，可通过 `--max-concurrency` 显式调整。

### Python ASGI 工厂

```python
from pii_zh.service import create_app

app = create_app(
    profile_version="community-presidio-bie73-cascade-v1",
    model_path="models/pii-zh-qwen3-0.6b-24class",
    primary_model_path="models/uer/cluener2020-primary",
    device="cpu",
)
```

```bash
uvicorn deploy_local:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1 \
  --no-access-log
```

## Presidio API

同一 pipeline 可以注册为 Presidio 中文识别来源：

```python
from pii_zh.full_bie73 import build_full_bie73_service_pipeline
from pii_zh.presidio import create_analyzer_engine

pipeline = build_full_bie73_service_pipeline(
    model_path="models/pii-zh-qwen3-0.6b-24class",
    primary_model_path="models/uer/cluener2020-primary",
    device="cpu",
)
analyzer = create_analyzer_engine(cascade_pipeline=pipeline)
results = analyzer.analyze(
    text="联系人张三，病历号 MR-2026-001。",
    language="zh",
)
```

Presidio entity 名称通过项目 taxonomy 映射。CLI、HTTP 与 Presidio adapter 共享同一优先级和
fusion 合同，避免不同入口产生不同的同类覆盖结果。

## 已验证结果

所有 F1 均为字符级 exact-span 匹配。冻结的
`native_presidio_2_2_363_cluener_cn_common_v5` closed-8 主检测基线为：

| split | strict micro F1 | strict macro F1 |
|---|---:|---:|
| Formal | `0.9527327902` | `0.8732108951` |
| Chat | `0.9361379310` | `0.8399701577` |
| Pooled | `0.9473637236` | `0.8667338443` |

这些数字属于 native Presidio 主检测器，不属于 BIE73 权重。实现合同在模型解码前屏蔽全部
closed-8 标签，原样保留 primary 输出，并让 primary 赢得跨分支 overlap，所以组合服务的
closed-8 投影与冻结 primary 分支完全相同。`0.9473637236` 不能称为整套 Open-24 服务 F1；
Open-24 增强服务尚无新的严格整体 F1，因此不声称增强服务严格超过基线，也不作 SOTA 声明。

其他服务回归：

| 数据集 | 结果 | 说明 |
|---|---:|---|
| CLUENER | strict micro F1 `0.794555` | 历史兼容分支；BIE73 无贡献 |
| CrossWOZ | `2` emitting docs / `2` spans | 对照服务 `8 / 8`；无 span-level gold，不报告 F1 |

## 数据与部署边界

BIE73 本轮数据共 88,000 条，全部为项目生成的合成文本：72,000 条 train 进入梯度训练，8,000 条
dev 与 8,000 条 independent test 不进入梯度。没有加入真实 PII、客户数据或公开 NER 训练行；
模型仍继承初始化模型及 Qwen3 backbone 的上游权重。

合成评测不能替代真实目标域验证。部署前应按实际 augmentation allow-list 评估误报、漏报、延迟
和资源占用，保留人工抽检，并避免在业务日志中保存输入原文或命中的敏感值。检测与 validator 只
给出候选跨度和格式证据，不判断某个标识是否真实签发、属于某人或必须依法脱敏。
