# pii-zh-qwen

面向简体中文的本地 PII 检测模型与级联服务：用 24 类中文 token-classification
模型理解上下文，用中文规则补充结构化标识，统一输出可用于检测与脱敏的字符级 span，
并提供 Presidio 兼容层。

[![Release](https://img.shields.io/github/v/release/whyiug/pii-detect-model?include_prereleases)](https://github.com/whyiug/pii-detect-model/releases/tag/v0.2.0rc1)
[![Model](https://img.shields.io/badge/Hugging%20Face-model-FFD21E)](https://huggingface.co/Forrest20231206/pii-zh-qwen3-0.6b-24class)
[![License](https://img.shields.io/github/license/whyiug/pii-detect-model)](LICENSE)

当前公开版本为 <code>v0.2.0rc1</code>。它提供可直接安装的 CLI、FastAPI 服务和公开模型权重，
适合本地评估、应用集成与二次开发。

## 主要能力

- 24 类简体中文 PII 模型，覆盖身份、联系方式、账号、设备与位置等常见信息；
- rules-only、model-only、cascade 三种运行模式；
- Presidio 兼容层、中文规则、格式校验、阈值与冲突合并；
- CLI 检测/脱敏、批量 JSONL、FastAPI 和本地 Docker；
- 默认本地运行；CLI 不会静默上传文本，也不会自动下载模型。

## 快速开始

需要 Python 3.10 或更高版本。当前包从 GitHub Release 安装：

~~~bash
python -m pip install \
  "pii-zh-qwen[cascade,service] @ https://github.com/whyiug/pii-detect-model/releases/download/v0.2.0rc1/pii_zh_qwen-0.2.0rc1-py3-none-any.whl"
~~~

### 1. 无模型快速检测

规则模式不下载权重，安装后即可运行：

~~~bash
pii-zh detect \
  --profile c1-conservative-v2 \
  --text '测试邮箱 demo@example.com，联系电话 13800138000。' \
  --pretty
~~~

脱敏：

~~~bash
pii-zh redact \
  --profile c1-conservative-v2 \
  --text '测试邮箱 demo@example.com，联系电话 13800138000。' \
  --replacement '<PII>'
~~~

检测结果包含实体类型、字符起止位置、置信分数和来源，不回显命中的原始片段。

### 2. 下载模型并启用级联

~~~bash
python -m pip install "huggingface_hub>=0.27"
hf download Forrest20231206/pii-zh-qwen3-0.6b-24class \
  --revision v0.2.0rc1 \
  --local-dir ./models/pii-zh-qwen3-0.6b-24class
~~~

~~~bash
pii-zh detect \
  --profile community-model-cascade-v1 \
  --mode cascade \
  --model-path ./models/pii-zh-qwen3-0.6b-24class \
  --device cpu \
  --text '联系人张三，手机号 13800138000，邮箱 demo@example.com。' \
  --pretty
~~~

如有 CUDA，可将 <code>--device cpu</code> 改为 <code>--device cuda</code>。服务不会替你选择、
占用或清理 GPU。

### 3. 启动本地 API

~~~bash
pii-zh serve \
  --profile community-model-cascade-v1 \
  --mode cascade \
  --model-path ./models/pii-zh-qwen3-0.6b-24class \
  --device cpu \
  --host 127.0.0.1 \
  --port 8000
~~~

~~~bash
curl --fail --silent --show-error \
  -H 'content-type: application/json' \
  --data '{"text":"测试邮箱 demo@example.com"}' \
  http://127.0.0.1:8000/v1/analyze
~~~

默认仅绑定 loopback，并限制请求体、文本长度、并发和超时。完整参数见
[部署指南](docs/cascade-deployment.md)和 [API 文档](docs/community-model-service.md)。

## 工作方式

~~~mermaid
flowchart LR
    A[输入文本] --> B[中文规则与格式校验]
    A --> C[24 类中文 PII 模型]
    B --> D[span 合并与冲突处理]
    C --> D
    D --> E[检测结果或脱敏文本]
~~~

规则擅长身份证号、银行卡号、邮箱等结构化标识；模型负责姓名、地址和上下文相关实体；
级联层对来源、阈值、格式和重叠 span 做统一处理。架构细节见
[级联设计](docs/cascade-architecture.md)。

## 支持的实体类型

- 身份：<code>PERSON_NAME</code>、<code>DATE_OF_BIRTH</code>、<code>CN_RESIDENT_ID</code>、
  <code>PASSPORT_NUMBER</code>、<code>DRIVER_LICENSE_NUMBER</code>、
  <code>SOCIAL_SECURITY_NUMBER</code>
- 联系与位置：<code>PHONE_NUMBER</code>、<code>EMAIL_ADDRESS</code>、<code>ADDRESS</code>、
  <code>GEO_COORDINATE</code>
- 金融与组织账号：<code>BANK_CARD_NUMBER</code>、<code>BANK_ACCOUNT_NUMBER</code>、
  <code>EMPLOYEE_ID</code>、<code>STUDENT_ID</code>、<code>MEDICAL_RECORD_NUMBER</code>
- 网络与设备：<code>WECHAT_ID</code>、<code>QQ_NUMBER</code>、<code>ALIPAY_ACCOUNT</code>、
  <code>USERNAME</code>、<code>IP_ADDRESS</code>、<code>MAC_ADDRESS</code>、
  <code>DEVICE_ID</code>
- 其他：<code>VEHICLE_LICENSE_PLATE</code>、<code>SECRET</code>

<code>SECRET</code> 用于发现安全敏感字符串，不一定属于法律意义上的个人信息。

## 效果概览

以下结果来自同一个冻结的 Open-24 协议；该协议为 100% 合成数据。模型轨与完整服务轨的输入、
后处理和比较器不同，不能交叉解读。

| 轨道 | 系统 | strict micro F1 | strict recall | PII-free 文档 FPR |
|---|---|---:|---:|---:|
| 单模型 | pii-zh-qwen3-0.6b-24class | **0.9867** | **0.9907** | **0.0000** |
| 单模型对照 | AIguard fast Open-24 | 0.2512 | 0.1862 | 0.1160 |
| 完整服务 | model + cn_common_v6 rules cascade | **0.9888** | **0.9781** | **0.0000** |
| 服务对照 | Presidio + CLUENER + cn_common_v6 | 0.6237 | 0.5185 | 0.1622 |

在公开 PII Bench ZH 的 pooled closed-8 测试上，本模型单独运行的 strict micro F1 为
0.5575，不是该榜单最佳结果；本项目的重点是可控、可集成的中文级联服务。历史 CLUENER
兼容回归中，非默认、post-hoc 兼容配置的 F1 为 0.7946，超过第一版服务的 0.7870；该结果
不是当前 Open-24 完整级联的直接评测。
评测协议、比较器和限制见 [benchmark landscape](docs/benchmark_landscape.md) 与
[技术指南](docs/aiguard24_full_model_and_cascade_technical_guide.md)。

## 模型与数据

- 基座/初始化：<code>ZJUICSR/AIguard-pii-detection-fast</code>，固定 revision
  <code>677a5ebc1600fef61e8973cafd3026be322b3a73</code>；
- 架构：Qwen3 token classification，596,100,145 个 FP32 参数；
- 输出：<code>O + 24×BIO = 49</code> 个 token 标签；
- 训练集 87,995 条，开发集 2,005 条；
- 两个 split 均为确定性合成/开源派生数据，未使用客户或生产 PII。

训练集与开发集共享 17 个模板家族，因此合成评测不能替代真实业务域验证。公开
<code>v0.2.0rc1</code> 模型可直接推理，但该版本没有分发冻结评测所用的 calibration bundle；
上面的级联指标是项目冻结回放结果，不应被描述为默认命令的逐字节复现结果。

## 使用边界

- 模型会漏检和误检；正式脱敏前应使用目标域数据重评阈值，并保留人工抽检与回滚能力。
- 不要把真实 PII、访问令牌或私钥提交到 issue、日志或公开测试样例。
- 本项目不是法律或合规意见，也不替代访问控制、数据最小化和人工审查。
- 若处理高风险数据，建议固定 Git/HF revision、离线保存依赖并记录部署配置。

## 文档

- [Hugging Face Model Card](https://huggingface.co/Forrest20231206/pii-zh-qwen3-0.6b-24class)
- [v0.2.0rc1 Release](https://github.com/whyiug/pii-detect-model/releases/tag/v0.2.0rc1)
- [级联部署](docs/cascade-deployment.md)
- [服务 API](docs/community-model-service.md)
- [模型数据集与训练技术报告](docs/model-training-technical-report.md)
- [训练与评测技术指南](docs/aiguard24_full_model_and_cascade_technical_guide.md)
- [安全问题报告](SECURITY.md)

## 从源码开发

~~~bash
git clone https://github.com/whyiug/pii-detect-model.git
cd pii-detect-model
uv sync --frozen --extra cascade --extra service --extra training --extra dev
uv run ruff check src tests scripts
uv run python -m scripts.run_current_community_rc_tests
~~~

模型权重、真实凭据、原始数据和本地运行目录不进入 Git。

## License

仓库代码采用 [Apache License 2.0](LICENSE)。模型、数据集和依赖的第三方条款独立适用，
再分发或生产使用前请阅读 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
