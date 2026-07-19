# Synthetic v1 数据物化

`synthetic_v1` 是公开权重训练池中的确定性程序生成数据。当前数据版本为 `1.3.0`，使用
模板资产 `1.2.0`、生成器 `2.4.0` 和物化器 `1.3.0`。它不读取真实个人数据，也不调用
网络或 GPU。模板资产经过本地人工筛选；实体值由版本化生成器构造，结构化类型仍必须通过
对应 validator。JSONL 仅写入外部数据目录，不提交 Git。

模板资产包含 87 个正例家族和 20 个 hard-negative 家族。其中 `SECRET` 至少覆盖部署访问
令牌、回调签名密钥、移动会话令牌、数据库密码、Bearer 鉴权头和私钥片段六种独立语义，
并包含引号、空格、方括号及中英文标点等左右边界。`MEDICAL_RECORD_NUMBER` 覆盖预约、
住院、病历更新、表格、出生登记、转诊、检验归档和出院结算八个独立家族。新增的构建任务
ID、配置键名、软件发布日期和设备产品型号家族均是明确无个人实体的 hard negative；它们只
用于训练相似表面下的语义边界，不得被重新解释为 PII 或真实凭据。

`1.3.0` 只增加 11 个 train-only 正例家族：`VEHICLE_LICENSE_PLATE` 3 个、
`EMPLOYEE_ID` 2 个、`DATE_OF_BIRTH` 2 个，以及 `DRIVER_LICENSE_NUMBER`、
`GEO_COORDINATE`、`MAC_ADDRESS`、`PASSPORT_NUMBER` 各 1 个。它们分别覆盖停车、年检、
高速通行、排班、门禁、年龄核验、会员权益、网约车资质、救援定位、零信任接入和跨境住宿，
并使用书名号、键值、括号、引号、斜线和 JSON 风格等不同左右边界。本轮没有新增 `SECRET`。

值格式通过模板占位符上的显式 `value_variants` opt-in：旧 96 个模板全部保持 `legacy`，新模板
才使用虚构车牌前缀、非生产部门工号、紧凑/点分出生日期、`DL-ZZ`/`PZ` 哨兵、带符号空格
坐标和本地管理的小写连字符 MAC。未知 label/variant 组合在资产加载时 fail closed；旧格式生成
方法没有改写。

## 稳定 family 分配

数据版本 `1.3.0` 使用 `stable-pinned-v3`，不再根据当前模板全集和随机种子重新计算 family
split。版本化资产
`src/pii_zh/data/synthetic/assets/stable_family_allocation_v3.json` 从已版本化的 v2 allocation 资产
继承并逐项固定全部 96 个 pre-v1.3 模板（train 65、validation 15、test 16），没有读取旧
JSONL 正文或实体值；任何旧 ID 的 family/split 漂移都会在导入时失败。v3 allocation 资产
SHA-256 为 `49b3c265991f2659ae3796f278da48361c440f57acbd91665df06616bf071647`。
旧 96 项在同 seed 下去除必然变化的版本/hash 绑定字段后，渲染与稳定 metadata 投影 SHA 为
`df4c63239ae319a287ed7ecaa966aa3fdbbdaeff3987f3ecdd6916b60de53b76`。

新增 11 个 family 逐项登记为 train-only，并按 3/2/2/1/1/1/1 的 label role 计数校验。
因此历史 96 项逐项不迁移，尤其 validation/test family 完全不变。模板 ID 未登记、family 漂移、
新增项不再是人工正例、role 与实体标签不符或计数变化时，加载和物化均 fail closed。

## 冻结 holdout

v1.3 物化器只重新生成 `train`。`validation.jsonl` 和 `test.jsonl` 从 v1.2 目录以只读文件描述符
逐块复制，复制过程中及落盘后都核验 bytes/SHA，绝不回退到重新生成，也不解析 JSONL 记录：

- parent manifest 文件 SHA：`e1bd5a6b094bdcd6566c94f5774313505924051d551d74c74617a21e309d023b`；
- validation：2,000 条、5,250,600 bytes、SHA `9affd99f5dd6a2aaac37b7975f3c3f051e7edb2d44be471a7238d27498c0b40d`；
- test：2,000 条、5,096,362 bytes、SHA `26e49949d0bd876b51c693b7b31135b5ce503ca6a5ab7cfca37177e0ccbd4384`。

为在不读取冻结正文的条件下验证新 train 的实体值隔离，仓库还钉住由 pre-v1.3 确定性代码和
legacy 资产内存重放得到的 6,832 个 holdout `entity_value_groups` 承诺，canonical SHA 为
`28d7eda8260a8f908defef2a36d4434500b0c3244aa5d62e9d7daf3988306ae3`。未来代码若不能复现该
承诺，物化直接失败，不能声称零碰撞。

新 train 不从 seed 的初始 value namespace 重新起步。物化器先用同一个
`SyntheticGenerator` 按 v1.2 的 train→validation→test 顺序完整预消耗 20,000 条父序列，
核验上述 6,832 承诺后，直接从该 generator 的 continuation 状态生成 16,000 条 v1.3 train。
因此 legacy value counter、随机 offset 和 document index 都从父序列末尾继续；manifest 的
`generation.value_namespace` 与 `splitting.entity_value_isolation` 明确记录
`v1.2-parent-sequence-continuation-v1`、预消耗记录数和零碰撞结果。默认 20,000 条合同测试会
实际构建该 train，并与 pinned holdout 承诺核对，禁止恢复为从 seed 起点重新生成。

## 构建

从仓库根目录运行；先将 `$DATA_ROOT` 设为外部数据根目录：

```bash
python scripts/materialize_synthetic_data.py \
  --holdout-source-dir "${DATA_ROOT}/processed/public_release_pool/synthetic_v1_2"
```

默认生成 20,000 条记录，种子为 `20260711`，输出到：

```text
$DATA_ROOT/processed/public_release_pool/synthetic_v1_3/
```

官方 CLI 不提供 `--count`、`--seed` 或 `--hard-negative-ratio` override；这些值与 v1.2
parent manifest、文件哈希和 value-group 承诺共同冻结。底层函数的可注入 contract 仅用于隔离的
单元测试，不能通过 CLI 替换官方 trust root。

CLI 固定执行以下门禁：

- `train/validation/test` 记录数严格按 `80/10/10` 分配；每份 hard negative 均约 30%；
- 按 `stable-pinned-v3` 审计资产应用模板家族分配，24 个 core label 在每份都出现；
- 正例和 hard-negative 家族各自只分配到一个 split；11 个新增语义家族仅进入 train；
- validation/test 必须与 v1.2 的固定 bytes/SHA 完全相同，任何父 manifest 或文件漂移都失败；
- 同一模板家族和同一实体值哈希不得跨 split，生成实体值在本批次中全局唯一；
- 每条记录按严格 schema、字符 offset 和数据池路由回读验证；
- 正文和实体值不得包含配置的合成捷径词；
- `train.jsonl`、`validation.jsonl`、`test.jsonl` 和 `dataset_manifest.json` 均设为
  `0444`，输出目录设为 `0555`；
- 已存在的目标目录绝不覆盖。需要重建时应使用新的版本目录并保留旧 manifest。

## Manifest

`dataset_manifest.json` 不含正文或实体值，只记录计划哈希、仅适用于 train 的生成器/模板资产、
family allocation 与 curation audit 哈希、随机种子、split 算法、历史固定/新增 train-only family
计数、记录计数、continuation value namespace 和每个文件的 bytes/SHA。`frozen_splits` 单独保存 v1.2 manifest、generation、
assets、provenance、coverage 与逐 split 复制 lineage，避免把 generator 2.4 错误归因给旧 holdout。
训练配置应引用
manifest 和具体输入文件哈希，而不是仅引用可变目录名。

实体值虽然具有格式真实性，但全部来自程序生成，并不来源于个人信息。身份证区域前缀、银行卡
IIN、`.invalid` 邮件域、RFC 文档 IP 范围和本地管理 MAC 前缀等安全约束记录在生成器与
provenance 中；不得把这些数据重新描述成真实金标或生产流量。
