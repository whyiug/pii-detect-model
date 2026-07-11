# Synthetic v1 数据物化

`synthetic_v1` 是公开权重训练池中的确定性程序生成数据。当前数据版本为 `1.2.0`，使用
模板资产 `1.1.0`、生成器 `2.3.0` 和物化器 `1.2.0`。它不读取真实个人数据，也不调用
网络或 GPU。模板资产经过本地人工筛选；实体值由版本化生成器构造，结构化类型仍必须通过
对应 validator。JSONL 仅写入外部数据目录，不提交 Git。

模板资产包含 76 个正例家族和 20 个 hard-negative 家族。其中 `SECRET` 至少覆盖部署访问
令牌、回调签名密钥、移动会话令牌、数据库密码、Bearer 鉴权头和私钥片段六种独立语义，
并包含引号、空格、方括号及中英文标点等左右边界。`MEDICAL_RECORD_NUMBER` 覆盖预约、
住院、病历更新、表格、出生登记、转诊、检验归档和出院结算八个独立家族。新增的构建任务
ID、配置键名、软件发布日期和设备产品型号家族均是明确无个人实体的 hard negative；它们只
用于训练相似表面下的语义边界，不得被重新解释为 PII 或真实凭据。

## 稳定 family 分配

数据版本 `1.2.0` 使用 `stable-pinned-v2`，不再根据当前模板全集和随机种子重新计算 family
split。版本化资产
`src/pii_zh/data/synthetic/assets/stable_family_allocation_v2.json` 固定了 87 个历史模板的原始
分配（train 56、validation 15、test 16）；这份快照只从旧 `synthetic_v1` 三份 JSONL 投影
读取 `provenance.template_id`、`template_group` 和 `split`，没有读取正文或实体值。资产
SHA-256 为 `755e4d77e18003afd82bbb30a91942022fe68a19acb9e5e310592fc39552984f`。

本次新增的 3 个 `SECRET`、2 个 `MEDICAL_RECORD_NUMBER` 和 4 个 hard-negative family 在资产
中逐项登记为 train-only。因此历史 train family 不迁移，历史 validation/test family 完全不变；
seed 只影响各 split 内的确定性值生成，不影响 family 分配。模板 ID 未登记、已登记 ID 的 family
改变、或新增项不再满足 3/2/4 语义角色时，加载和物化均 fail closed，必须先审阅并版本化更新
allocation 资产。

## 构建

从仓库根目录运行：

```bash
python scripts/materialize_synthetic_data.py
```

默认生成 20,000 条记录，种子为 `20260711`，输出到：

```text
/data1/datasets/pii-detect-model/processed/public_release_pool/synthetic_v1/
```

CLI 固定执行以下门禁：

- `train/validation/test` 记录数严格按 `80/10/10` 分配；每份 hard negative 均约 30%；
- 按 `stable-pinned-v2` 审计资产应用模板家族分配，24 个 core label 在每份都出现；
- 正例和 hard-negative 家族各自只分配到一个 split；9 个新增语义家族仅进入 train；
- 同一模板家族和同一实体值哈希不得跨 split，生成实体值在本批次中全局唯一；
- 每条记录按严格 schema、字符 offset 和数据池路由回读验证；
- 正文和实体值不得包含配置的合成捷径词；
- `train.jsonl`、`validation.jsonl`、`test.jsonl` 和 `dataset_manifest.json` 均设为
  `0444`，输出目录设为 `0555`；
- 已存在的目标目录绝不覆盖。需要重建时应使用新的版本目录并保留旧 manifest。

## Manifest

`dataset_manifest.json` 不含正文或实体值，只记录计划哈希、生成器/物化器版本、模板资产、
family allocation 资产与 curation audit 哈希、provenance 快照哈希、随机种子、split 算法
版本、历史固定/新增 train-only family 计数、记录计数、标签/领域/质量级覆盖、模板家族分配
哈希，以及每个 JSONL 文件的字节数和 SHA-256。训练配置应引用
manifest 和具体输入文件哈希，而不是仅引用可变目录名。

实体值虽然具有格式真实性，但全部来自程序生成，并不来源于个人信息。身份证区域前缀、银行卡
IIN、`.invalid` 邮件域、RFC 文档 IP 范围和本地管理 MAC 前缀等安全约束记录在生成器与
provenance 中；不得把这些数据重新描述成真实金标或生产流量。
