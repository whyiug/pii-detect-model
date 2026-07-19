# 中文 PII 级联本地 HTTP 服务

HTTP 适配层复用 `CascadePipeline`，不会另写一套检测、validator 或融合逻辑。零参数工厂默认
`rules-only`，不会加载模型、访问 Hugging Face Hub、调用外部 API、启用遥测或记录请求正文。

## 安装与本机启动

```bash
pip install "pii-zh-qwen[service]"
uvicorn --factory pii_zh.service.app:create_app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1 \
  --no-access-log
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

模型级联需要同时安装两个 extra：

```bash
pip install "pii-zh-qwen[cascade,service]"
```

模型必须先下载到本地并完成项目 manifest/hash 检查。应用启动文件应显式构造 pipeline：

```bash
: "${MODEL_ROOT:?set MODEL_ROOT to the local model-storage root}"
```

```python
import os
from pathlib import Path

from pii_zh.cascade import CascadeConfig, CascadePipeline
from pii_zh.service import create_app

model_path = (
    Path(os.environ["MODEL_ROOT"]) / "pii-detect-model/releases/release-model"
).resolve(strict=True)
pipeline = CascadePipeline.from_pretrained(
    model_path,
    config=CascadeConfig(mode="cascade"),
    device="cpu",
    local_files_only=True,
)
app = create_app(pipeline=pipeline)
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

2026-07-15 的有界本地 smoke 中，首次冷依赖构建在 `519.96s` 内完成；排除 root README、本机
bytecode 和旧 `egg-info` 后，最终缓存重建为 `3.41s`；BuildKit 当次增量 context 传输为 `8.69kB`，
完整 inference allowlist 的未压缩 regular-file 载荷为 `2,861,521` bytes，临时 image 为 207MB。
预期 context 与最终序列化 layers 对构建机私有路径以及 `$REPO_ROOT`、`$MODEL_ROOT` 所代表的本地
存储前缀扫描均为 0 命中。
随后真实运行默认 CLI，并把短时 HTTP 容器仅发布到 host `127.0.0.1` 的随机高端口。`/healthz` 和
`/v1/analyze` 均通过，analyze 响应不含命中原值；临时容器与 image 已清理。该结果只证明默认
rules-only 容器路径可构建、可运行，不证明 Presidio 容器、本地模型、GPU parity、性能或发布门已经
完成。完整构建指纹和清理状态见 [`../docker/README.md`](../docker/README.md)。

### 显式运行车牌 successor

镜像默认命令继续保留历史 v1。`c1-conservative-v2` 必须显式选择：

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

2026-07-16 的 successor-specific 验收还从同一规则实现真实启动了显式 v2 HTTP app；服务只在
容器内部 `127.0.0.1` 监听，容器使用 `--network=none` 且没有宿主端口，验证后已清理。wheel
侧另在隔离环境通过 Presidio 路径；上述 rules-only image 本身不安装 Presidio、Torch、
Transformers 或模型权重。精确 wheel/image 哈希、四路径结果和仍未关闭的发布门保留在内部历史验收
`successor_rules_only_packaging_container_acceptance_v1`。
内部 `successor_rules_only_service_source_closure_v2`
另行绑定完整 packaged source、四条服务路径、lint policy/evidence、CI、packaging contract 和 v1
ancestry；本地 GitHub-quality 同范围已通过，但新增文件尚未进入 Git/托管 CI。
其中原工作树工程验收覆盖 4 个 valid + 6 个 hard negative；successor wheel 安装后 smoke 是
3 个 valid + 1 个 invalid document，容器 smoke 是 1 个 valid + 1 个 invalid document。后两者
证明打包/运行路径没有丢失既有候选语义，不是扩大后的质量样本，也不能估计 recall 或 FPR。

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

## C4 CPU 测试覆盖与未完成门

2026-07-16 已额外完成一次从 wheel 安装的隔离环境验收：在仓库目录外安装
`pii-zh-qwen[service,presidio]==0.1.0`，并以 CPU/offline 方式通过 Python、已安装 CLI、
Presidio `AnalyzerEngine` 和 FastAPI `TestClient` 四条 rules-only 路径。wheel SHA、环境版本、
无原文断言和剩余边界由内部 `clean_wheel_service_acceptance_v1` 验收记录绑定。
该 clean-wheel rules-only 收据本身没有安装或运行 `[cascade]` 模型依赖，不能替代真实
checkpoint、GPU parity 或
发布候选验收。

随后又在 `include-system-site-packages=false` 的独立 venv 实际安装同一 wheel 的 `[cascade]`
extra；pip report 绑定 `requested_extras=["cascade"]` 与 wheel SHA，`pip check`、installed
site-packages import 均通过。使用一个本地已有、manifest-bound 但 `release_eligible=false` 的
开发 checkpoint，CPU/offline 的 model-only、cascade、redaction 以及 socket-connect
fail-closed smoke 通过。完整版本、哈希和限制由内部
`clean_wheel_cascade_acceptance_v1` 验收记录绑定。
它本身只关闭安装/运行 smoke，不关闭默认模型、质量、许可、GPU parity 或发布门。后续
`legacy_cluener_10` 的 G2-v3 actual-service/paired compatibility subgate 已由独立
[closure](../reports/evaluation/legacy_cluener_g1_g2_threshold_050_v3_comparison_v2.json) 闭合，但该公开
dev、post-result/nonblind 结果仍不改变这里的 production/release 边界。

当前离线合成测试覆盖以下运行时合同：

| 维度 | CPU/合成证据 | 状态 |
|---|---|---|
| batch/single | predictor、Qwen recognizer 和 `CascadePipeline` 逐 span parity；HTTP 当前只有单文本 API | 已覆盖 |
| CPU | rules-only HTTP 真实执行；tiny safetensors predictor 在 CPU 实际加载；模型 pipeline 的 `device="cpu"` 参数绑定 | 已覆盖，非发布 checkpoint 性能证据 |
| 长文本 | tokenizer-aware 多窗口、跨窗 center ownership，以及长 Unicode HTTP 请求 | 已覆盖 |
| 重叠 | 实体冲突 fusion 与跨窗重复候选只保留一次 | 已覆盖 |
| offset | emoji、全角字符和换行之前的 Python 左闭右开字符 offset，并验证 `text[start:end]` | 已覆盖 |
| 替换 | 多 span 右到左替换，以及 HTTP 从原始字符 offset 精确替换 | 已覆盖 |
| 失败回退 | recognizer/context/validator/timeout 均返回无原文通用失败；模型失败不静默降级 | 已覆盖 fail-closed 语义 |

`cascade` 模式中的模型分支失败时，服务故意返回通用失败，不会在同一请求里悄悄改用规则结果。若部署方
接受降级，应在受信任控制面显式切换到一个独立的 `rules-only` pipeline，并记录模式变化；这条规则回退
不等价于模型结果，也不能用来掩盖模型故障。

**GPU parity 仍未完成。** CPU 合成测试不能证明真实冻结 checkpoint 在 CPU/GPU 上逐 span、score、
batch/single 完全一致。该门只能在重新查询共享机器 GPU、选择当时空闲卡且不处理其他进程后，用同一
冻结 checkpoint、tokenizer、配置和输入集真实运行；在收据落盘前不得把 C4 总门标为完成。
