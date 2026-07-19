# Community 24 类模型级联服务

`community-model-cascade-v1` 是一个**非默认、显式启用**的社区 profile。它把
`cn_common_v6` 中文规则、结构化校验器和项目 24 类本地 token-classification 模型装入同一条
`CascadePipeline`；模型适配器使用 `QwenPiiRecognizer`，因此同一 pipeline 也可注册到 Presidio
`AnalyzerEngine`。零参数默认仍是 `c1-conservative-v1` 的 `rules-only`，不会加载模型。

## 安装和模型边界

```bash
pip install "pii-zh-qwen[cascade,service]"
```

模型参数必须先由部署者放到本地目录。运行时不接受 Hugging Face 仓库 ID、URL、revision 或
`trust_remote_code`；加载固定使用 `local_files_only=True`、`trust_remote_code=False` 和
`use_safetensors=True`，并验证完成状态的 `training_manifest.json` 及其文件哈希。community 工厂还会在
导入 PyTorch 前验证以下精确合同：

- `manifest_type=aiguard24_community_training_v1`、`status=completed`、AIGuard24 固定来源；
- `qwen3_bi` + `pii_attention_mode=full` + SDPA，共享 state-dict 转换没有新增或丢弃参数；
- 恰好 `O + 24 × B/I = 49` 个标签，顺序与打包 taxonomy 完全一致；
- 已合并的单一 `model.safetensors`、tokenizer/config 和 manifest 哈希互相绑定；
- validation checkpoint receipt 完整，但当前不硬编码尚未产生的跨 seed selection hash。

causal、JPT、MacBERT、普通 Qwen3、错误/多余 BIO 标签、pickle 权重、adapter-only 目录、被修改的
manifest 或权重都会终止启动，不会静默退回 rules-only。

公开 service profile 的合法组合被冻结为：

| profile | rules-only | model-only | cascade |
|---|---:|---:|---:|
| `c1-conservative-v1` | 是 | 否 | 否 |
| `c1-conservative-v2` | 是 | 否 | 否 |
| `community-model-cascade-v1` | 否 | 是 | 是 |

## CLI 检测与脱敏

```bash
MODEL=/absolute/path/to/pii-zh-24-model

pii-zh detect \
  --profile community-model-cascade-v1 \
  --mode cascade \
  --model-path "$MODEL" \
  --device cpu \
  --calibration /absolute/path/to/calibration.json \
  --threshold PERSON_NAME=0.70 \
  --text '测试邮箱 demo@example.com'

pii-zh redact \
  --profile community-model-cascade-v1 \
  --mode cascade \
  --model-path "$MODEL" \
  --text '测试邮箱 demo@example.com' \
  --replacement '<PII>'
```

## 本地 HTTP API

打包 CLI 的 `serve` 命令只接受 `127.0.0.1`、`::1` 或 `localhost`，固定单 worker 且关闭 access
log：

```bash
pii-zh serve \
  --profile community-model-cascade-v1 \
  --mode cascade \
  --model-path "$MODEL" \
  --device cpu \
  --calibration /absolute/path/to/calibration.json \
  --max-concurrency 4 \
  --host 127.0.0.1 \
  --port 8000
```

启动后使用 `/healthz`、`/v1/analyze` 和 `/v1/redact`。三个响应都报告 profile；健康、检测和脱敏
响应中的 `model_identity` 只含 manifest SHA、架构、attention、taxonomy/label schema，以及可选
calibration/threshold 策略 SHA，不含模型路径、输入文本或实体值。请求中的模型异常返回通用 500；
同一请求不会只返回规则分支结果。`serve` 支持显式 `--max-concurrency`；未指定时 CPU 为 4，任何
`cuda*` device 为 1，避免同一模型被默认并发挤占显存。

Python 工厂等价写法：

```python
from pathlib import Path

from pii_zh.cascade import COMMUNITY_MODEL_SERVICE_PROFILE_VERSION
from pii_zh.service import create_app

app = create_app(
    profile_version=COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
    mode="cascade",
    model_path=Path("/absolute/path/to/pii-zh-24-model"),
    device="cpu",
    micro_batch_size=4,
    calibration=Path("/absolute/path/to/calibration.json"),
    thresholds={"PERSON_NAME": 0.70, "ADDRESS": 0.65},
    max_concurrency=4,
)
```

## Presidio 接入

```python
from pathlib import Path

from pii_zh.cascade import build_community_model_service_pipeline
from pii_zh.presidio import create_analyzer_engine

pipeline = build_community_model_service_pipeline(
    Path("/absolute/path/to/pii-zh-24-model"),
    mode="cascade",
    device="cpu",
    calibration=Path("/absolute/path/to/calibration.json"),
    thresholds={"PERSON_NAME": 0.70},
)
analyzer = create_analyzer_engine(cascade_pipeline=pipeline)
results = analyzer.analyze(text="测试邮箱 demo@example.com", language="zh")
```

## 完整 24 类路由与校验边界

community profile 没有缩小模型 taxonomy：24 类全部打开模型分支并有逐类端到端 fixture。四个语义
类别 `PERSON`、`CN_ADDRESS`、`DATE_OF_BIRTH`、`USERNAME` 走
`model → semantic threshold → fusion → output`；其余二十类走
`model → named deterministic validator → threshold → fusion → output`。

结构化校验器覆盖手机号、邮箱、身份证、护照、驾驶证、社保号、银行卡、银行账户、车牌、工号、
学号、病历号、微信、QQ、支付宝、IP、MAC、设备号、坐标和 SECRET。每个 grammar 都有仓库合成
sentinel fixture；适用时也覆盖常见真实格式。validator 只证明“格式与模型标签相容”，不证明号码已
签发、属于某人或必须脱敏。租户策略和语义仍是独立边界。`c1-conservative-v2` 的严格车牌行为没有
被改写；只有 community profile 额外接受 `岚…` 合成 sentinel。

没有显式校准时，community 路由保留语义短串保护，新开放结构化模型分支使用 0.50 基础阈值。传入
validation-fitted calibration 或 threshold 后，标签会严格归一到 core24 输出名，未知标签、重复的
raw/Presidio alias、NaN 和越界阈值都失败关闭；recognizer 成为唯一阈值权威，route 不再叠加隐藏
阈值。因此正式评估和部署可复用同一个 `build_community_model_service_pipeline` builder。输入的原始
校准值不出现在 API，只输出规范化策略的 SHA。

因此这是可复现的 community/BYO-model 集成 profile，不是默认生产 profile，也不应把完整级联指标
写成 standalone 模型指标。完整 24 类“可达”是实现合同，不等于每类精度已经由真实世界 hidden set
证明。是否改为默认仍须由独立的服务净收益、PII-free FPR、hidden set 和资源门决定。
