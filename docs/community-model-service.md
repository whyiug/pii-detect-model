# Community 24 类模型级联服务

`community-model-cascade-v1` 是一个**非默认、显式启用**的社区 profile。它把
`cn_common_v6` 中文规则、结构化校验器和项目 24 类本地 token-classification 模型装入同一条
`CascadePipeline`；模型适配器使用 `QwenPiiRecognizer`，因此同一 pipeline 也可注册到 Presidio
`AnalyzerEngine`。零参数默认仍是 `c1-conservative-v1` 的 `rules-only`，不会加载模型。

## 安装和模型边界

```bash
python -m pip install \
  "pii-zh-qwen[cascade,service] @ https://github.com/whyiug/pii-detect-model/releases/download/v0.2.0rc1/pii_zh_qwen-0.2.0rc1-py3-none-any.whl"

MODEL="$PWD/models/pii-zh-qwen3-0.6b-24class"
hf download Forrest20231206/pii-zh-qwen3-0.6b-24class \
  --revision v0.2.0rc1 \
  --local-dir "$MODEL"
```

下载和运行是两个独立步骤：服务只接受本地模型目录，不接受 Hugging Face 仓库 ID、URL 或
revision。加载固定使用 `local_files_only=True`、`trust_remote_code=False` 和
`use_safetensors=True`，并在导入 PyTorch 前检查模型文件、manifest、架构和标签合同：

- `manifest_type=aiguard24_community_training_v1`、`status=completed`、AIGuard24 固定来源；
- `qwen3_bi` + `pii_attention_mode=full` + SDPA，共享 state-dict 转换没有新增或丢弃参数；
- 恰好 `O + 24 × B/I = 49` 个标签，顺序与打包 taxonomy 完全一致；
- 已合并的单一 `model.safetensors`、tokenizer/config 和 manifest 哈希互相绑定。

causal、JPT、MacBERT、普通 Qwen3、错误/多余 BIO 标签、pickle 权重、adapter-only 目录、被修改的
manifest 或权重都会终止启动，不会静默退回 rules-only。

可用的 service profile 与运行模式组合为：

| profile | rules-only | model-only | cascade |
|---|---:|---:|---:|
| `c1-conservative-v1` | 是 | 否 | 否 |
| `c1-conservative-v2` | 是 | 否 | 否 |
| `community-model-cascade-v1` | 否 | 是 | 是 |

## CLI 检测与脱敏

```bash
pii-zh detect \
  --profile community-model-cascade-v1 \
  --mode cascade \
  --model-path "$MODEL" \
  --device cpu \
  --text '测试邮箱 demo@example.com'

pii-zh redact \
  --profile community-model-cascade-v1 \
  --mode cascade \
  --model-path "$MODEL" \
  --text '测试邮箱 demo@example.com' \
  --replacement '<PII>'
```

如有基于本地验证数据拟合的校准文件，可额外传入 `--calibration /path/to/calibration.json`；
也可以用可重复的 `--threshold LABEL=VALUE` 覆盖逐类阈值。

## 本地 HTTP API

打包 CLI 的 `serve` 命令只接受 `127.0.0.1`、`::1` 或 `localhost`，固定单 worker 且关闭 access
log：

```bash
pii-zh serve \
  --profile community-model-cascade-v1 \
  --mode cascade \
  --model-path "$MODEL" \
  --device cpu \
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
    model_path=Path("models/pii-zh-qwen3-0.6b-24class"),
    device="cpu",
    micro_batch_size=4,
    max_concurrency=4,
)
```

## Presidio 接入

```python
from pathlib import Path

from pii_zh.cascade import build_community_model_service_pipeline
from pii_zh.presidio import create_analyzer_engine

pipeline = build_community_model_service_pipeline(
    Path("models/pii-zh-qwen3-0.6b-24class"),
    mode="cascade",
    device="cpu",
)
analyzer = create_analyzer_engine(cascade_pipeline=pipeline)
results = analyzer.analyze(text="测试邮箱 demo@example.com", language="zh")
```

## 完整 24 类路由与校验边界

community profile 不会缩小模型 taxonomy：24 类全部启用模型分支。四个语义
类别 `PERSON`、`CN_ADDRESS`、`DATE_OF_BIRTH`、`USERNAME` 走
`model → semantic threshold → fusion → output`；其余二十类走
`model → named deterministic validator → threshold → fusion → output`。

结构化校验器覆盖手机号、邮箱、身份证、护照、驾驶证、社保号、银行卡、银行账户、车牌、工号、
学号、病历号、微信、QQ、支付宝、IP、MAC、设备号、坐标和 SECRET。validator 只证明“格式与
模型标签相容”，不证明号码已签发、属于某人或必须脱敏；具体处理策略仍由部署方决定。

没有显式校准时，community 路由保留语义短串保护，新开放结构化模型分支使用 0.50 基础阈值。传入
validation-fitted calibration 或 threshold 后，标签会严格归一到 core24 输出名，未知标签、重复的
raw/Presidio alias、NaN 和越界阈值都失败关闭；recognizer 成为唯一阈值权威，route 不再叠加隐藏
阈值。CLI、HTTP 服务和 Python 集成都复用同一个
`build_community_model_service_pipeline` builder。输入的原始校准值不出现在 API，只输出规范化
策略的 SHA。

这是可复现的 community/BYO-model 集成 profile；零参数默认保留轻量的 rules-only 模式。
完整级联指标不能写成 standalone 模型指标，部署前仍应使用目标领域数据评估误报、漏报、延迟和
资源占用。
