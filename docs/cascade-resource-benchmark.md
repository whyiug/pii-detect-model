# Cascade CPU / rules-only 资源诊断

[`scripts/benchmark_cascade.py`](../scripts/benchmark_cascade.py) 是发布
`CascadePipeline` 的最小离线资源 smoke。它固定使用
`CascadeConfig(mode="rules-only")` 和 `CascadePipeline.detect_batch`，不会接受模型目录、GPU 设备或远程
输入，也不会隐式下载权重。

这个脚本生成的是 `DIAGNOSTIC_ONLY` 报告，**不是发布性能结论、容量承诺或生产 SLA**。内置样例很小，
本地 canonical JSONL 的长度与分布也不一定代表生产流量。正式 C6 资源报告仍需冻结硬件、CPU 配额、数据
revision、并发模型和模型 checkpoint，并单独覆盖 GPU/model/cascade 轨。

## 运行

使用内置的 example 域名和 IANA 文档保留地址做快速 smoke：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/benchmark_cascade.py \
  --builtin-synthetic \
  --output reports/cascade-resource/rules-only-smoke.json \
  --batch-size 32 \
  --warmup 3 \
  --repetitions 10
```

也可以读取已经存在的本地 canonical JSONL：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/benchmark_cascade.py \
  --input /local/path/workload.canonical.jsonl \
  --output reports/cascade-resource/rules-only-local.json \
  --batch-size 32 \
  --warmup 3 \
  --repetitions 20
```

`--input` 与 `--builtin-synthetic` 必须二选一。输入必须是普通本地文件；脚本拒绝 symlink、空 workload、
超过聚合限制的 workload 和非 canonical 行。`--output` 必须不存在。报告先在目标目录写入临时文件并
`fsync`，然后以不可覆盖的硬链接原子发布，最终权限为只读；并发创建同名输出时也会 fail closed。

## 指标口径

- `startup.first_pipeline_initialization`：当前 Python 进程中第一次构造 rules-only pipeline 的时间；不包含
  解释器启动和模块导入。
- `startup.cold_process_start`：显式为 `unavailable`。可靠的进程冷启动需要外部 supervisor，以及受控的
  page cache/filesystem cache；脚本不会把一次进程内计时冒充冷启动。
- `warmup`：完整 workload 的预热轮数、文档数、耗时和聚合 emitted span 数；不进入热路径统计。
- `hot_path.latency_milliseconds`：预热后每次 batch 调用、按 batch size 归一化的每文档调用，以及每个完整
  workload pass 的 p50/p95。p50/p95 对全部重复样本做线性插值。
- `hot_path.throughput`：完整 pass 的端到端 wall time 口径 `docs/s`、`chars/s`；包含 batch 循环和聚合
  emitted span 计数，但不包含预热与 pipeline 构造。
- `model_call_rate`：rules-only 配置的结构性结果，分子固定为 0，分母是计时阶段处理的文档数。它不是通过
  猜测日志得到的值。
- `cpu.max_rss_after`：进程级 high-water RSS，包含解释器、导入和 pipeline；不是 pipeline 独占分配量。
  Linux 上将 `ru_maxrss` 的 KiB 转成 bytes。报告同时保留测量前 high-water 和前后进程 CPU time。

## 证据和隐私边界

报告绑定完整 `CascadeConfig`、configuration SHA-256、脚本及直接运行时组件的逐文件 SHA-256 和合并
implementation SHA-256，并记录 Python、OS/kernel、CPU 数量/affinity 和计时器。输入只记录 source kind、
内容 SHA-256、文档数和字符数。

输出和成功/失败终端消息不包含：

- 原文或实体值；
- document ID；
- 输入文件名、绝对路径、当前工作目录或 Python 可执行文件路径；
- 单文档或单实体结果。

canonical JSONL 原文仍会在当前进程内存中被发布 pipeline 处理，因此应在受控的本地环境运行；“聚合
输出”不等于可以把未经授权的数据交给该脚本。输入 hash 也应按数据治理要求保存。

## 不可外推的内容

单次报告受共享服务器调度、CPU affinity、频率、缓存状态、Python build、workload 长度和 batch size
影响。它没有测量进程级冷启动、并发尾延迟、GPU、VRAM、checkpoint 加载、artifact 大小、模型调用或
HTTP 网络栈。跨机器或跨 workload 比较之前必须先固定这些条件；不能用本 smoke 支持“最快”“生产可用”
或完整级联系统的资源声明。
