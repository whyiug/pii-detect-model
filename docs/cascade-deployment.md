# 中文 PII 级联本地 HTTP 服务

HTTP 适配层复用 `CascadePipeline`，不会另写一套检测、validator 或融合逻辑。零参数工厂默认
`rules-only`，不会加载模型、访问 Hugging Face Hub、调用外部 API、启用遥测或记录请求正文。

## 安装与本机启动

```bash
python -m pip install \
  "pii-zh-qwen[service] @ https://github.com/whyiug/pii-detect-model/releases/download/v0.2.0rc1/pii_zh_qwen-0.2.0rc1-py3-none-any.whl"

pii-zh serve \
  --host 127.0.0.1 \
  --port 8000
```

默认绑定地址为 `127.0.0.1`。不要在含真实 PII 的共享机器上把服务直接绑定到
`0.0.0.0`；如确需跨容器访问，应另加认证、TLS、网络访问控制和明确的生命周期管理。

## API

以下示例只使用文档保留域名：

```bash
curl --fail-with-body http://127.0.0.1:8000/healthz

curl --fail-with-body \
  -H 'content-type: application/json' \
  -d '{"text":"测试邮箱 demo@example.com"}' \
  http://127.0.0.1:8000/v1/analyze

curl --fail-with-body \
  -H 'content-type: application/json' \
  -d '{"text":"测试邮箱 demo@example.com","replacement":"<EMAIL>"}' \
  http://127.0.0.1:8000/v1/redact
```

`/v1/analyze` 只返回 offset、类型、分数、来源和无原文决策轨迹。`/v1/redact` 会返回完整脱敏
文本；未识别部分仍可能包含敏感信息，调用方必须继续按 PII 数据保护响应。

## 本地模型或完整级联

模型级联需要安装 `cascade` 和 `service` 两个 extra，并先把公开模型下载到本地目录：

```bash
python -m pip install \
  "pii-zh-qwen[cascade,service] @ https://github.com/whyiug/pii-detect-model/releases/download/v0.2.0rc1/pii_zh_qwen-0.2.0rc1-py3-none-any.whl"

MODEL_ROOT="$PWD/models/pii-zh-qwen3-0.6b-24class"
hf download Forrest20231206/pii-zh-qwen3-0.6b-24class \
  --revision v0.2.0rc1 \
  --local-dir "$MODEL_ROOT"
```

服务只从本地加载 safetensors，并在启动时检查 manifest、文件哈希、架构和 49 标签合同。应用启动
文件可以显式选择完整级联 profile：

```python
import os
from pathlib import Path

from pii_zh.cascade import COMMUNITY_MODEL_SERVICE_PROFILE_VERSION
from pii_zh.service import create_app

model_path = Path(os.environ["MODEL_ROOT"]).resolve(strict=True)
app = create_app(
    profile_version=COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
    mode="cascade",
    model_path=model_path,
    device="cpu",
    max_concurrency=4,
)
```

将其保存为 `deploy_local.py` 后运行：

```bash
uvicorn deploy_local:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

服务不会接收模型仓库 ID 或 URL，因此不存在请求触发下载的路径。

## 轻量 rules-only 容器边界

默认 inference Docker profile 只安装 base package 与 `service` extra，直接运行与 Python/CLI
相同的 rules pipeline；它不为了默认 HTTP 路径引入 Presidio/spaCy，也不包含 torch、Transformers、
CUDA 或模型权重。Presidio 仍是 wheel 的显式可选集成；本地模型容器必须显式选择
`RUNTIME_PROFILE=local-model`。

### 选择严格车牌规则

`c1-conservative-v2` 提供更严格的车牌检测行为，需显式选择：

```bash
docker build \
  --build-arg RUNTIME_PROFILE=rules-only \
  -f docker/inference.Dockerfile \
  -t pii-zh-qwen-inference:local .

printf '%s' '登记号牌为京A12345' | docker run --pull=never --rm -i \
  --runtime=runc \
  --network=none \
  --user app \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=16m \
  --pids-limit=64 \
  --memory=256m \
  --cpus=1 \
  --cap-drop=ALL \
  --security-opt=no-new-privileges:true \
  --env CUDA_VISIBLE_DEVICES= \
  pii-zh-qwen-inference:local \
  pii-zh detect --mode rules-only \
  --profile c1-conservative-v2 \
  --entity CN_VEHICLE_LICENSE_PLATE
```

## 默认边界与调优

零参数工厂采用以下保守限制：

- 请求体最大 256 KiB；同时检查 `Content-Length` 和实际接收字节数；
- 单条文本最多 100,000 个 Python 字符；
- pipeline 调用超时 30 秒；
- 最多 4 个 pipeline 调用并行执行；
- 错误响应只返回通用原因，不回显请求正文；
- 所有 API 响应使用 `Cache-Control: no-store`，应用本身不写输入日志。

可以在自建启动模块中向 `create_app` 传入更小的
`max_request_body_bytes`、`max_text_chars`、`request_timeout_seconds` 和 `max_concurrency`。
超时不会强制终止正在执行的 Python/native 推理线程；该线程结束前仍占用并发槽，避免超时请求不断
叠加后台推理。生产部署还应在反向代理和容器层设置连接超时、内存/CPU/GPU 限额。

## 隐私与日志

应用没有输入日志和遥测，但部署平台、代理、APM、shell history 及调用方仍可能记录数据。保持固定
API 路径，不把 PII 放入 URL/query string，并关闭代理的 request-body capture。若需要审计，只记录
请求 ID、耗时、状态码和聚合计数；不要记录正文、命中值、认证头或完整异常。

## 运行时行为

| 维度 | 行为 |
|---|---|
| batch/single | Python predictor 和 `CascadePipeline` 支持批量与单条输入；HTTP API 当前接收单条文本 |
| 长文本 | 模型推理使用 tokenizer-aware 多窗口，并对跨窗候选去重 |
| 重叠 | 冲突融合后保留确定性的字符覆盖结果 |
| offset | 输出使用 Python 左闭右开字符 offset，可用 `text[start:end]` 取回原片段 |
| 替换 | 多 span 从右到左替换，保持原始字符位置稳定 |
| 错误处理 | recognizer、context、validator 或超时异常返回不含输入原文的通用错误 |

`cascade` 模式中的模型分支失败时，服务故意返回通用失败，不会在同一请求里悄悄改用规则结果。若部署方
接受降级，应在受信任控制面显式切换到一个独立的 `rules-only` pipeline，并记录模式变化；这条规则回退
不等价于模型结果，也不能用来掩盖模型故障。

CPU 与 GPU 的浮点结果可能存在细微差异。正式部署前应在所选设备、批大小和目标文本分布上验证
span、分数、吞吐和显存占用。
