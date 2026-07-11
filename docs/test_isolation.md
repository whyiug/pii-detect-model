# 测试集隔离策略 v1.0

状态：**隔离策略已定义，测试集尚未下载或构建**。本文件不声称 `pii-bench-zh`、内部
金标、客户 holdout 或现网 ERNIE 结果已经可用。

## 1. 不可跨越的边界

`evaluation_only` 中任何文本、标签、模板、统计量或模型响应均不得用于：

- 模型训练、继续预训练、蒸馏或伪标签筛选；
- teacher prompt、候选类型描述或生成模板开发；
- tokenizer/规则/上下文词表开发；
- calibration、阈值、融合器或 early stopping；
- 主动学习优先级和错误驱动的数据生成；
- 人工选择 checkpoint 或训练超参数。

仅在候选 recipe、阈值和 checkpoint 已由 train/validation 数据冻结后运行测试。测试结果
只用于最终比较和发布门禁；观察到测试失败后修改系统必须形成新实验世代，并使用独立验证
证据，不能反复调到测试集通过。

## 2. 物理和权限隔离

- 评测数据放在 `data/processed/evaluation_only/` 或独立只读存储，不与训练 glob 共目录。
- 训练配置只能声明 `public_release_pool` 或经授权的 `private_enterprise_pool`；检测到
  `evaluation_only`/`quarantined` 路径、pool 字段或 manifest ID 时立即失败。
- 冻结 manifest 记录文件哈希、来源 revision、记录数、许可和访问组。自动评测以只读方式
  挂载；训练服务账号无读取权限是优先方案。
- 测试明文、逐条预测和错误样本只保存在受控位置。公开报告仅给聚合指标、置信区间和
  合成示例。
- 任何临时副本、notebook cache、CI artifact 和本地下载都遵守同一保留/删除策略。

## 3. 分组隔离

在赋予 split 前，对 tenant、会话、文档、模板家族、生成 seed 系列、归一化实体值哈希、
来源/时间窗口以及 MinHash/SimHash 近重复簇做联合分组。同组记录只进入一个 split。

建议冻结测试套件：

1. `internal_iid_test`：同领域的新文档；
2. `tenant_holdout_test`：从未用于训练/验证的客户组；
3. `domain_holdout_test`：未见业务场景；
4. `time_holdout_test`：冻结点之后的新时间桶；
5. `hard_negative_test`；
6. `noise_stress_test`；
7. `long_document_test`；
8. `pii_bench_zh_formal`；
9. `pii_bench_zh_chat`。

内部集合需要合同与数据授权，默认不公开。外部 `pii-bench-zh` 仅在固定 revision、许可证
核验和 manifest 生成后进入评测池，formal/chat 分开报告。

## 4. 变更控制

每个评测集有唯一 `dataset_id`、语义版本和不可变内容哈希。允许的修订仅包括经记录的标签
错误修复、法律删除请求或确定性格式迁移；每次修订发布新版本，保留差异计数和影响报告。
禁止静默替换样本或覆盖旧 manifest。

接触过冻结测试明文/标签的人员不得把记忆中的具体样本直接写入训练模板、规则或 prompt。
若泄漏无法排除，污染的测试集降级为开发集并创建新的独立测试集。

## 5. 执行门禁

评测入口必须验证：

- 候选 checkpoint、配置、校准器、阈值和规则 revision 已冻结；
- 测试 manifest 哈希与批准记录一致；
- 训练数据 manifest 与测试组/近重复/entity-value hash 无交集；
- 运行不访问公网、不回写测试集、不记录原文；
- 逐实体、逐域、最差组、hard-negative 和端到端指标均带样本量与 95% 置信区间；
- 至少保存模型单独、规则单独、现网 ERNIE（若真实可用）及组合系统的可比版本信息；
- 任何缺失基线明确标为 `not_available`，不得填零或构造结果。

## 6. 当前未满足项

- 尚无现网 ERNIE + rules 的模型级或端到端基线；
- 尚无内部金标、跨客户、跨时间或无 PII 冻结集；
- 尚未下载并冻结 `pii-bench-zh`；
- 尚无模型 checkpoint、阈值或校准器可评测。

因此阶段 A 的“现网基线报告”和业务/安全签字仍是开放交付，后续实现不得把本策略文档
误报为退出条件已经完成。

