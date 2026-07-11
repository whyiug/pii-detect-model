# 统一字符 Span 数据合同 v1.0

状态：**规范已定义，数据尚未导入**。本合同不证明 OpenPII、内部金标、客户数据或任何
教师输出已经通过质量/许可门禁。

## 1. 规范目标

所有数据源先转换为 UTF-8 JSONL。每行是一份独立文档；字符 span 是唯一规范真值，BIO、
token offsets、窗口和 logits 都是可重建的派生产物。原始 PII 不得提交到 Git。

## 2. 文档记录

必填字段：

| 字段 | 类型 | 约束 |
|---|---|---|
| `schema_version` | integer | 当前为 `1` |
| `doc_id` | string | 稳定 UUID 或不暴露原文的受控哈希标识 |
| `text` | string | 解码后的规范原文，不得在日志中输出 |
| `entities` | array | 可为空；每项符合第 3 节 |
| `language` | string | 简体中文主线使用 `zh-Hans-CN` |
| `domain` / `scene` | string | 版本化词表；未知值显式写 `unknown` |
| `source_id` / `source_revision` | string | 指向来源注册表和不可变 revision |
| `quality_level` | string | `G0/S0/S1/S2/U/N` |
| `quality_gate` | boolean | 是否通过该等级所需质量门禁 |
| `public_weight_training_allowed` | boolean | 必须来自明确许可结论，禁止推断 |
| `license` | string | SPDX 标识或受控的合同/评审引用 |
| `pool` | string | 四个允许池之一 |
| `split` | string/null | 入池和分组去重后赋值 |

用于泄漏隔离的字段必须存在，可无适用值但不得默默省略：`tenant_group`、
`conversation_group`、`document_group`、`template_group`、`generator_seed_group`、
`entity_value_groups`、`near_duplicate_group`、`created_at_bucket`。组值使用不可逆、带命名空间
的稳定标识，不保存明文实体。

可选字段包括 `generator_version`、`teacher_annotation_ids`、`noise_types`、`review`、
`attributes`。新增字段必须向后兼容或提升 `schema_version`。

## 3. Entity 记录

每个实体必须含：

| 字段 | 类型 | 约束 |
|---|---|---|
| `start` / `end` | integer | `0 <= start < end <= len(text)`，左闭右开 |
| `label` | string | taxonomy 已声明标签 |
| `text` | string | 必须严格等于 `document.text[start:end]` |
| `source` | string | `gold/synthetic/rule/teacher/...` 受控值 |
| `risk_tiers` | array[string] | 必须是该标签允许的风险层；辅助标签为空 |
| `annotation_id` | string | 文档内唯一，支持裁决和 provenance |

原始层允许嵌套和重叠。转换器另存平面视图及冲突报告，绝不重写本记录。教师置信度、
规则校验结果和人工裁决状态放入结构化 `attributes`，不能冒充金标来源。

以下仅演示结构，值是不可用于真实联系/认证的合成占位文本：

```json
{"schema_version":1,"doc_id":"synthetic-demo-0001","text":"联系人<<PERSON_1>>，电话<<PHONE_1>>。","entities":[{"start":3,"end":15,"label":"PERSON_NAME","text":"<<PERSON_1>>","source":"synthetic_placeholder","risk_tiers":["T1"],"annotation_id":"e1"},{"start":18,"end":29,"label":"PHONE_NUMBER","text":"<<PHONE_1>>","source":"synthetic_placeholder","risk_tiers":["T1"],"annotation_id":"e2"}],"language":"zh-Hans-CN","domain":"documentation","scene":"contract_example","source_id":"repo_authored_placeholder","source_revision":"1","quality_level":"S0","quality_gate":true,"public_weight_training_allowed":true,"license":"Apache-2.0","pool":"public_release_pool","tenant_group":null,"conversation_group":null,"document_group":"synthetic-demo-0001","template_group":"contract-placeholder-v1","generator_seed_group":null,"entity_value_groups":[],"near_duplicate_group":null,"created_at_bucket":"2026-Q3","generator_version":"placeholder-only-v1","split":null}
```

占位符样本只适合合同/流水线测试，不能用于宣称模型质量。

## 4. 质量等级

| 等级 | 定义 | 建议权重 |
|---|---|---:|
| `G0` | 双标或裁决后的真实金标 | 1.00 |
| `S0` | 确定性合成且全部校验器通过 | 0.90–1.00 |
| `S1` | 至少两个独立教师一致，quote/offset/规则通过 | 0.70–0.90 |
| `S2` | 单教师高置信且校验通过 | 不高于 0.60 |
| `U` | 教师分歧、边界或类型不确定 | 0，转人工/候选蒸馏 |
| `N` | 经确认的困难负样本 | 0.50–1.00 |

质量和公开资格是两个正交门禁。`quality_gate=true` 不表示允许公开训练；许可允许也不表示
质量合格。

## 5. 四类数据池

| `pool` | 训练资格 | 公开权重资格 | 典型内容 |
|---|---|---|---|
| `public_release_pool` | 是 | 是 | 明确许可的公共/合成样本 |
| `private_enterprise_pool` | 仅获批私有训练 | 否 | 内部/客户授权数据 |
| `evaluation_only` | **绝不训练** | 不适用 | 外部基准、冻结测试集 |
| `quarantined` | 否 | 否 | 来源、许可、质量或隐私未决 |

进入公开池必须同时满足：`quality_gate=true`、`public_weight_training_allowed=true`、来源
revision 已冻结、许可证/条款已复核、PII 泄漏扫描通过、分组去重完成。任何未知值默认进入
`quarantined`。池之间移动必须生成审计记录，禁止复制后绕过原记录。

## 6. 切分和近重复隔离

禁止普通逐句随机切分。以下任一组相同的样本不得跨训练/验证/测试：tenant、会话、文档、
模板家族、生成 seed 系列、归一化实体值哈希、来源时间窗口和 MinHash/SimHash 近重复簇。
测试结构至少区分 IID、tenant holdout、domain holdout、time holdout、hard negative、noise、
long document、`pii-bench-zh` formal/chat。

先确定冻结测试和组分配，再拟合 tokenizer 以外的任何数据统计、teacher prompt、规则、
calibrator 或阈值。重复检测结果、split manifest 和算法版本必须持久化。

## 7. 来源与教师 provenance

每个 `source_id` 必须指向机器可读注册项：类型、revision、许可证/合同版本、署名、允许用途、
隐私分类、评审人/工单和条款快照哈希。每个教师标注另记录模型/服务 revision、部署方式、
prompt hash、解码配置、输入类别、校验结果和可用于公开权重的明确结论。

外部 API 默认不得进入公开池。真实 PII 不得发送给第三方 API。输出所有权、基础模型许可
或“服务不用于训练”声明均不能替代公开派生权重的专项许可判断。

## 8. 校验和日志

入池前至少校验：UTF-8、字段类型、span 范围/等式、标签/风险层、annotation ID 唯一性、
重叠统计、来源 revision、许可门禁、池规则、组字段、split 隔离、精确/近重复和禁止字段。

流水线日志仅允许记录 manifest ID、哈希、计数、比例、错误码和经批准的合成 canary。
异常不得打印 `text`、entity 值、API prompt/response、token、cookie 或客户规则。原始数据、
checkpoint、LoRA、optimizer state、梯度和缓存均按敏感产物管理。

## 9. Manifest 与可复现性

每次数据构建生成不可变 manifest：输入/输出文件哈希、记录数、每标签/域/质量级计数、
转换器 commit、配置哈希、随机种子、去重/切分算法版本、拒绝原因、来源注册表快照及执行环境。
发布报告可公开计数和哈希，不公开真实文本或私有路径。

