# pii-zh community v2 RC1 发布说明（待发布稿）

> 状态：GitHub 源码仓库已公开，Private Vulnerability Reporting 已启用但尚未完成独立账号实测；
> Hugging Face 仓库仍为 private，尚未上传模型。本文件是发布说明草稿，不能作为模型已经发布或
> 可用于生产的证明；许可证人工审批由独立自哈希回执证明。

## 本地候选身份

- 版本：`pii-zh-qwen 0.2.0rc1`
- 模型名：`pii-zh-qwen3-0.6b-24class`
- 模型任务：简体中文 PII token classification，24 个实体类型、49 个 BIO token 标签
- 应用形态：单模型，以及 Presidio、中文规则、中文 NER 与该模型组成的级联服务
- 冻结回执路径：
  `$RELEASE_RUN/community-cascade-release-v2.receipt.v2r3.json`
- 回执状态：`READY_FOR_USER_AUTHORIZATION`
- 回执内 `receipt_sha256`：
  `087b1c0e1b918e2c927c0653f312d8d0dc2954b6631b46fa22f9d7e194a84880`
- 回执文件 SHA-256：
  `5248a9746ce0676856cb29ecdd183f232a6078cd4b94e7138c539fa5bc1675e7`
- 回执同时记录 `local_candidate_complete=true`、`blocker_ids=[]`。这只说明本地 community-v2
  合同闭合，不等于 GitHub/Hugging Face 已发布，也不关闭许可证和生产门。

## 本 RC 包含什么

- 冻结的 seed-97 24 类模型候选；
- 字符级 span 推理 API 与 Transformers 兼容模型文件；
- Presidio、规则、中文 NER 与模型的级联服务实现；
- 训练、校准、比较器、质量聚合、独立重放、wheel、容器、SBOM 与本地验收证据；
- 100% 合成训练/验证数据的来源和限制说明。

Hugging Face 只能上传重新生成并重新验收的 `$PUBLICATION_DIR`。现有
`$RELEASE_RUN/model-package-v2r2` 是本地预授权包，不能原样上传。GitHub Release 可以附带最终
wheel、SBOM、checksums 与发布回执，但不附带 `wheelhouse/`，也不把模型权重提交到 GitHub
源码仓库。

## 可公开描述的合成 Open24 结果

以下结果仅适用于冻结、具名、100% 合成的 Open24 同协议评测。质量聚合使用 10,000 次按文档
配对 bootstrap；它不代表真实业务、人类盲测、跨域泛化或生产性能。

| 轨道 | 系统 | strict-span micro F1 | strict-span micro recall | PII-free document FPR |
|---|---|---:|---:|---:|
| 单模型 | 本候选 | 0.9867 | 0.9907 | 0.0000 |
| 单模型 | `aiguard_pii_detection_fast_open24_model_raw_v1` | 0.2512 | 0.1862 | 0.1160 |
| 集成服务 | 本候选 | 0.9888 | 0.9781 | 0.0000 |
| 集成服务 | `native_presidio_cluener_cn_common_v6_open24_v1` | 0.6237 | 0.5185 | 0.1622 |

差值均为“本候选减具名比较器”：
| 轨道 | Δ micro F1（95% CI） | Δ macro F1（95% CI） | Δ document FPR（95% CI） |
|---|---:|---:|---:|
| 单模型 | 0.7354 [0.7260, 0.7446] | 0.8583 [0.8537, 0.8631] | -0.1160 [-0.1253, -0.1069] |
| 集成服务 | 0.3651 [0.3570, 0.3733] | 0.5294 [0.5253, 0.5336] | -0.1622 [-0.1731, -0.1513] |

允许的表述应完整保留限定条件，例如：

> 在本项目冻结的具名合成 Open24 同协议评测中，候选单模型和集成服务分别取得 0.9867 和
> 0.9888 strict-span micro F1；这些结果不代表真实世界、跨域或生产表现。

不得把这些数字改写成“首个”“first”“SOTA”“最好”“最强”“行业领先”或
“production-ready”，也不得暗示合规保证。

## PII Bench ZH：仅作限制披露

PII Bench ZH 已被项目读取，是 100% 程序化合成的公开测试集，且没有 PII-free 文档。下表是
冻结后的 seed-97 `model_raw` 单次 posthoc 描述性结果；`public_test_exposed=true`、
`posthoc_lineage=true`、`selection_allowed=false`，并且没有通过历史描述性 active-comparator
envelope。它不参与模型、阈值或路由选择，也不能和 Open24 数字混算。

| suite | documents | strict micro F1 | strict macro F1 | 完整级联服务 |
|---|---:|---:|---:|---|
| Formal | 5,000 | 0.59921050 | 0.57994984 | N/A（未评测） |
| Chat | 3,000 | 0.47628738 | 0.41874684 | N/A（未评测） |
| Pooled | 8,000 | 0.55750532 | 0.52965259 | N/A（未评测） |

不得补造完整服务结果，不得据此声称相对其他模型领先，也不得把公开测试已见结果写成独立盲测。

## 数据与适用边界

训练集 87,995 条、开发验证集 2,005 条，均为确定性合成数据；训练/验证中的 PII-free 文档分别
为 39,600 和 900 条。训练资产不含客户数据或生产 PII。train/validation 存在 17 个
template-group overlap，因此开发结果也不是独立真实数据泛化证据。

本 RC 面向社区研究、复现和集成实验。部署方必须按自己的域、标签定义、阈值、误报/漏报代价和
法律义务进行人工评估；模型输出不能直接替代法律判断、合规控制或访问授权。

## 发布前仍需关闭的事项与已完成审批

- human/legal 许可证清单已由维护者 `whyiug` 于 2026-07-20 审批，无例外；冻结的机械报告仍保留
  pre-approval 状态，外部回执负责绑定最终文件；
- Hugging Face namespace 已固定为 `Forrest20231206`，private model repo 已创建但未上传模型；
- GitHub 源码仓库已公开，release-prep commit `565ceafd82f4c1001a0c4bf36e6ffc493f4e6e71`
  的 hosted CI 已通过；任何后续最终源码 commit 仍须重新运行并绑定成功记录；
- GitHub 私密安全报告渠道已启用，仍须由无管理权限账号完成纯合成端到端实测并生成回执；
- 必须从 `model-package-v2r2` 生成 publication-successor，替换旧 `SECURITY.md`，清除
  `unpublished_local_candidate`/preauthorization 发布语义，重新生成 checksums 并完成加载、
  安全与公开工件扫描；
- 必须由发布回执双向绑定 GitHub source commit/签名 tag 与 Hugging Face immutable commit。

逐项操作、硬门和命令模板见 [`PUBLISHING.md`](PUBLISHING.md)。
