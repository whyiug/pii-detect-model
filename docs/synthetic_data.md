# Synthetic v1 数据物化

`synthetic_v1` 是公开权重训练池中的确定性程序生成数据。它不读取真实个人数据，也不调用
网络或 GPU。模板资产经过本地人工筛选；实体值由版本化生成器构造，结构化类型仍必须通过
对应 validator。JSONL 仅写入外部数据目录，不提交 Git。

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
- 先在模板家族层做三路分组分配，24 个 core label 在每份都出现；
- 同一模板家族和同一实体值哈希不得跨 split，生成实体值在本批次中全局唯一；
- 每条记录按严格 schema、字符 offset 和数据池路由回读验证；
- 正文和实体值不得包含配置的合成捷径词；
- `train.jsonl`、`validation.jsonl`、`test.jsonl` 和 `dataset_manifest.json` 均设为
  `0444`，输出目录设为 `0555`；
- 已存在的目标目录绝不覆盖。需要重建时应使用新的版本目录并保留旧 manifest。

## Manifest

`dataset_manifest.json` 不含正文或实体值，只记录计划哈希、生成器/物化器版本、模板资产与
curation audit 哈希、provenance 快照哈希、随机种子、split 算法版本、记录计数、标签/领域/
质量级覆盖、模板家族分配哈希，以及每个 JSONL 文件的字节数和 SHA-256。训练配置应引用
manifest 和具体输入文件哈希，而不是仅引用可变目录名。

实体值虽然具有格式真实性，但全部来自程序生成，并不来源于个人信息。身份证区域前缀、银行卡
IIN、`.invalid` 邮件域、RFC 文档 IP 范围和本地管理 MAC 前缀等安全约束记录在生成器与
provenance 中；不得把这些数据重新描述成真实金标或生产流量。
