# 简体中文 PII 检测模型完整技术方案

> 文档版本：v1.3  
> 调研基线日期：2026-07-10  
> 目标任务：基于 Qwen 系列的简体中文 Token Classification / NER 模型，并与 Presidio、现有 ERNIE NER 和客户自定义规则协同工作  
> 状态：开源发布与教师策略定稿（API 条款复核）  
> 说明：本文是工程技术方案，不构成法律意见。实施前应由法务和数据合规人员复核实体范围、数据授权、保留期限、API 合同与适用标准。

---

## 0. 结论与推荐路线

本方案以“最终在 Hugging Face 发布模型、在 GitHub 发布训练与 Presidio 集成代码”为硬约束。推荐路线如下：

1. **公开学生模型采用双规格**：
   - 默认高性价比检查点：`Qwen/Qwen3-0.6B-Base`。
   - 可选质量检查点：`Qwen/Qwen3-1.7B-Base`；只有端到端 Presidio 指标显著更好且部署成本可接受时发布。
   - 两者均采用 Base 而非 Instruct 检查点，任务定义为 Token Classification。
   - 主架构为 `Qwen3BiForTokenClassification`：把因果注意力改为 Full Attention / Bidirectional Attention，并输出字符级 PII span。

2. **教师模型必须分层，不能只设一个“大 LLM 教师”**：
   - 低成本消融教师：`Qwen/Qwen3-4B-Base`；`Qwen3-1.7B-Base` 主要作为质量学生或流水线调试模型。
   - **默认判别式教师：`Qwen/Qwen3-8B-Base` + Full Attention + Token Classification**。
   - 性能上界教师：`Qwen/Qwen3-14B-Base` + Full Attention；只有蒸馏后的学生相对 8B 教师方案仍有稳定收益时才保留。
   - **MoE 研究上界：`Qwen/Qwen3-30B-A3B-Base` + Full Attention + Token Classification**。它是约 30.5B 总参数、每 token 约 3.3B 激活的 MoE，并非 30B dense；只在 8B/14B 已完成后做成本受控的 A/B，不作为默认教师。[R45][R46][R47]
   - **默认生成式标注/仲裁教师：自托管 `Qwen/Qwen3.6-27B` 的 text-only 服务**；高吞吐候选为 `Qwen/Qwen3.6-35B-A3B`，兼容性回退为 `Qwen/Qwen3-30B-A3B-Instruct-2507`。[R22][R23][R38][R39][R40]
   - 判别教师输出 token logits；生成教师只输出可校验的字符 span、候选标签和不确定性。二者采用不同的蒸馏通道。

3. **为什么默认判别教师选 8B，而不是无限增大**：
   - 0.6B 与 1.7B 的容量差距有限，1.7B 不再作为正式默认教师。
   - 8B/14B 属于 Qwen3 Base dense 系列，适合改造成同任务的 Token Classifier；但**不能先验假设 tokenizer 文件完全相同**，必须比较 tokenizer 文件 hash、词表、special tokens 和 offset 行为后，才能使用 token-logit KL。
   - 14B 的价值由“蒸馏后学生”而不是教师自身 F1 决定：若 14B 相对 8B 不能改善学生 Strict span F1、Tier-0 Recall 或最差域表现，就停止扩容。
   - 若仍需更大的同任务教师，优先实验 `Qwen3-30B-A3B-Base` MoE，而不是使用不存在于官方 Qwen3 Base 列表中的 `Qwen3-32B-Base`。MoE 分支需额外验证路由负载、专家塌缩、Full-Attention 改造和训练稳定性。[R45][R46][R47]
   - Qwen3.6 大模型采用不同的混合注意力/多模态后训练架构，更适合作为生成式标注器，而不是首版超大 Full-Attention Token Classifier。[R38][R39][R40]

4. **任务形式与训练方法**：
   - 训练标签采用 BIO；原始数据始终保存字符级左闭右开 span。
   - 主方案：Full Attention + 监督 Token Classification。
   - 推荐增强：双向领域继续适配、边界辅助头、多教师蒸馏、不确定性感知伪标签、噪声一致性训练。
   - JPT/Sequence Repetition 保留为不修改基础注意力结构的备选和正确性对照。[R3][R4][R5][R6]

5. **数据方案**：
   - 公共数据只做 warm-up；最终质量依靠自建中文合成数据、真实业务金标、困难负样本和主动学习。
   - `OpenPII 1.5M` 的 `zh-CN` 子集可做公开 warm-up，但必须重做 tokenizer 对齐与人工抽检。[R11]
   - `pii-bench-zh` 固定为外部冻结测试集，不参与训练、阈值选择、教师提示词开发或伪标签筛选。[R12]
   - 建立两条完全隔离的数据链：`public_release_pool` 只含可公开派生权重的来源；`private_enterprise_pool` 可含经合同授权的内部金标或客户适配数据，但默认不能用于公开权重。

6. **LLM API 可以用，但不应是公开模型的默认教师来源**：
   - API 最适合做候选 span 生成、冲突仲裁、困难样本解释和人工标注辅助，通常不能提供 token logits，因此不能替代 8B/14B 判别教师。
   - OpenAI 当前标准服务协议禁止使用输出开发与 OpenAI 产品和服务竞争的 AI 模型；其分类器例外又要求相关模型不得向第三方分发或商业提供。公开条款并未自动判定简体中文 PII Token Classifier 是否属于“竞争模型”，因此不能自行推导为允许：面向 Hugging Face 开源，应取得覆盖训练与分发的书面确认，否则该来源不进入 `public_release_pool`。[R28]
   - Anthropic 的公开说明一方面列出 specialized classifier / information extraction 为可接受用途，另一方面又要求书面许可并列出“把输出作为训练目标”为禁止项；面向开源发布应按“需书面确认”处理。[R29]
   - Gemini API 存在竞争模型限制；免费服务还会将输入输出用于产品改进并可能由人工审核，且明确不应提交敏感、机密或个人信息。付费服务的数据处理改善不等于获得训练并开源派生权重的授权。[R30]
   - 阿里云百炼声明不会将客户数据用于模型训练，但模型体验服务的生成内容有用途、商业化和第三方分发限制；不得把“体验服务”输出作为公开训练数据，正式付费 API 仍需逐份审核合同。[R31][R43][R44]
   - 最稳妥路径是优先使用自托管 Apache-2.0 Qwen 开放权重教师；外部 API 只有通过法律、隐私和来源门禁后才进入 `public_release_pool`。

7. **Presidio 集成**：
   - 初期保留现有 ERNIE，与 Qwen、结构化规则和客户规则并行。
   - 优先使用 Presidio 的 `HuggingFaceNerRecognizer`；Full Attention、自定义解码、分实体阈值和校准由 `QwenPiiRecognizer` 实现。[R9][R10]
   - 结构化 PII 由正则、格式解析和校验位兜底；Qwen 主检姓名、详细地址、上下文账号和非标准表达；ERNIE 作为通用 PERSON/LOCATION/ORG 辅助。
   - 客户和场景差异保存在版本化策略配置中，不硬编码进公共模型权重。

8. **开源发布策略**：
   - Hugging Face：至少发布 `zh-pii-qwen3-0.6b-bi`；可选发布 `zh-pii-qwen3-1.7b-bi`，并可把教师仅作为研究 artifact 或完全不发布。
   - GitHub：发布训练、数据转换、评测、蒸馏、Presidio 插件、规则示例、CI 和可复现实验配置。
   - 可选 Hugging Face Dataset：只发布许可兼容的公共/合成数据与 Dataset Card，不发布任何真实客户数据。
   - 权重使用 `safetensors`；自定义架构通过 `auto_map` 发布，短期需要 `trust_remote_code=True`，长期争取向 Transformers 上游贡献实现，降低采用门槛与远程代码风险。[R32][R33][R34]

9. **最终选型规则**：
   - 默认候选：`Qwen3-0.6B Full-Attention + 金标监督 + 8B 判别教师蒸馏 + 自托管 Qwen3.6 生成标注 + Presidio 融合`。
   - 质量档候选：把同一训练配方应用到 `Qwen3-1.7B-Base`，用真实端到端收益决定是否公开。
   - 14B 相对 8B 只有在至少 3 个种子上使蒸馏学生 Strict span F1 提升建议不少于 0.2–0.3 个百分点，且 Tier-0 Recall、hard-negative 和最差域均不退化时保留。
   - 商业 API 教师未经专项书面授权，不进入公开训练数据或公开权重 provenance。

---

## 1. 项目目标、边界与成功定义

### 1.1 目标

构建一个面向中国大陆简体中文文本的 PII 检测系统，覆盖：

- 客服对话、即时通信和语音转写文本；
- 表单、合同、物流单、医疗和金融文本；
- 应用日志、JSON、键值文本、Markdown、邮件和工单；
- 中英文混排、全半角、空格/横线分隔、OCR/ASR 噪声；
- 客户自定义账号、单号、工号及行业实体。

系统必须同时满足：

- 字符级精确定位；
- 高风险实体的高召回；
- 可配置的过度脱敏控制；
- Presidio 兼容；
- 可解释、可回滚、可按客户和场景配置；
- 训练数据来源、版本、许可和模型版本可追溯。

### 1.2 非目标

第一阶段不追求：

- 仅靠一个模型替代所有规则和校验器；
- 解决任意嵌套实体和关系抽取；
- 判断某条公开信息是否在法律意义上必然需要脱敏；
- 以公开合成集上的高 F1 直接作为生产上线依据；
- 用生成式输出替代字符 span 分类。

### 1.3 推荐的上线成功门槛

以下是工程建议门槛，应由安全、业务和法务共同确认：

| 指标 | 建议门槛 |
|---|---:|
| 内部金标集：模型严格字符 span Micro-F1 | ≥ 94% |
| 高风险结构化实体：端到端 Recall | ≥ 99.5% |
| 姓名、地址、联系方式：端到端 Recall | ≥ 98% |
| 任一关键客户/场景相对现网召回下降 | 不超过 0.5 个百分点 |
| 无 PII 文档误报率 | ≤ 2% |
| 端到端 PII 字符泄漏率 | ≤ 0.2% |
| 非 PII 字符过度遮盖率 | ≤ 1% |
| p95 延迟 | 不高于现网 ERNIE 方案的 1.5 倍，或满足业务 SLA |

必须同时报告 95% 置信区间，不能只报告单个点估计。

---

## 2. 法规和标签范围的工程化解释

《个人信息保护法》第四条采用“与已识别或者可识别自然人有关”的广义定义，并排除匿名化后的信息；敏感个人信息包括生物识别、特定身份、医疗健康、金融账户、行踪轨迹和不满十四周岁未成年人的个人信息等。[R15][R16]

因此，系统应分为三层，而不是把所有内容都塞进一个平面模型标签集：

1. **检测层**：识别稳定、可标注的原子 span。
2. **风险层**：给实体分级，决定不同召回和误报成本。
3. **策略层**：结合客户、场景、用途、上下文、允许列表和规则，决定是否遮盖、如何遮盖。

GB/T 35273-2020 仍可作为个人信息安全工程参考，但其官方页面显示 2026-01-28 的复审结论为“修订”；实施时需重新确认是否已有替代版本。去标识化效果评估可参考 GB/T 42460-2023。[R17][R18]

---

## 3. 标签体系设计

### 3.1 设计原则

- 模型标签表示“文本中是什么”，客户配置表示“该场景要不要处理”。
- 高风险结构化标识尽量使用具体类别，不统一压成 `ID`。
- 不把所有日期都当生日，不把所有组织和城市都当 PII。
- 模型训练标签与 Presidio entity type 解耦，通过映射层连接。
- 原始标注支持嵌套；Token Classification 训练视图按照确定性规则压平成非重叠 BIO 序列。

### 3.2 v1 核心模型标签

建议首版稳定在 20–25 个核心实体，避免一开始扩到过多低频类别：

| 模型标签 | 含义 | 主要检测器 | 风险层 |
|---|---|---|---|
| `PERSON_NAME` | 自然人姓名、别名 | Qwen + ERNIE | T1 |
| `PHONE_NUMBER` | 手机、座机、分机 | 规则 + Qwen | T1 |
| `EMAIL_ADDRESS` | 邮箱 | 规则 + Qwen | T1 |
| `ADDRESS` | 可定位个人的详细地址 | Qwen + 地址规则 | T1 |
| `DATE_OF_BIRTH` | 出生日期 | Qwen + 上下文 | T1 |
| `CN_RESIDENT_ID` | 大陆居民身份证号 | 校验规则 + Qwen | T0 |
| `PASSPORT_NUMBER` | 护照号 | 规则 + Qwen | T0 |
| `DRIVER_LICENSE_NUMBER` | 驾驶证号 | 规则 + Qwen | T0 |
| `SOCIAL_SECURITY_NUMBER` | 社保/医保个人编号 | 规则 + Qwen | T0 |
| `BANK_CARD_NUMBER` | 银行卡号 | Luhn/格式 + Qwen | T0 |
| `BANK_ACCOUNT_NUMBER` | 银行账户号 | 客户规则 + Qwen | T0 |
| `VEHICLE_LICENSE_PLATE` | 车牌 | 规则 + Qwen | T1 |
| `EMPLOYEE_ID` | 员工编号 | 客户规则 + Qwen | T1/T2 |
| `STUDENT_ID` | 学号 | 客户规则 + Qwen | T1/T2 |
| `MEDICAL_RECORD_NUMBER` | 病历、就诊、住院编号 | 行业规则 + Qwen | T0 |
| `WECHAT_ID` | 微信号 | Qwen + 规则 | T1 |
| `QQ_NUMBER` | QQ 号 | Qwen + 规则 | T1 |
| `ALIPAY_ACCOUNT` | 支付宝账号 | Qwen + 规则 | T0/T1 |
| `USERNAME` | 个人用户名/登录名 | Qwen | T1 |
| `IP_ADDRESS` | IPv4/IPv6 | 规则 + Qwen | T1/T2 |
| `MAC_ADDRESS` | MAC 地址 | 规则 | T1/T2 |
| `DEVICE_ID` | IMEI、设备序列号等 | 规则 + Qwen | T1 |
| `GEO_COORDINATE` | 精确经纬度 | 规则 + Qwen | T0/T1 |
| `SECRET` | 密码、API Key、Token、私钥片段 | 规则 + Qwen | T0，安全敏感而非纯 PII |

`CN_UNIFIED_SOCIAL_CREDIT_CODE`、企业名称、公司地址和公共客服电话默认不作为个人 PII；它们可以作为客户敏感信息规则包开启。

### 3.3 v1.1 扩展敏感属性

只有在获得足够金标数据和明确业务需求后，再增加：

- `HEALTH_INFORMATION`
- `FINANCIAL_INFORMATION`
- `BIOMETRIC_INFORMATION`
- `RELIGIOUS_BELIEF`
- `PRECISE_MOVEMENT`
- `MINOR_INFORMATION`
- `AGE`
- `GENDER`
- `OCCUPATION`

这些标签边界和场景依赖明显，应单独评测和配置，不能因法规列举就默认把文本中所有相关词语全部遮盖。

### 3.4 辅助/混淆标签

建议做一个实验分支，在训练标签空间加入下列“明确非 PII 或场景混淆”标签，输出时忽略：

- `ORDER_ID`
- `TRANSACTION_ID`
- `INVOICE_CODE`
- `PRODUCT_CODE`
- `ORGANIZATION`
- `JOB_TITLE`
- `GENERIC_DATE`
- `PUBLIC_HOTLINE`
- `ORG_ADDRESS`

目的不是输出这些实体，而是让模型对“身份证号 vs 订单号”“个人地址 vs 公司地址”“生日 vs 普通日期”获得显式监督。是否保留由消融实验决定。

### 3.5 嵌套和冲突的压平规则

原始数据允许多个 span；训练 BIO 序列按以下优先级压平：

1. 高风险、格式明确的具体标识；
2. 具体账号类型；
3. 姓名、地址、联系方式；
4. 扩展敏感属性；
5. 辅助混淆标签。

若两个 span 重叠：

- 完全相同边界时保留更具体类别；
- 包含关系时，训练视图保留高优先级 span；
- 原始嵌套信息仍保存在数据中，供 Presidio 规则和端到端评测使用；
- 禁止静默丢弃，转换报告必须统计冲突数和压平结果。

---

## 4. 总体系统架构

```text
                        ┌──────────────────────┐
原始文本 ──预处理/切窗──▶│ Qwen PII Recognizer │──┐
                        └──────────────────────┘  │
                                                  │
                        ┌──────────────────────┐  │
                        │ 现有 ERNIE Recognizer│──┼──▶ 结果校准与融合 ──▶ Presidio Anonymizer
                        └──────────────────────┘  │
                                                  │
                        ┌──────────────────────┐  │
                        │ 通用/行业/客户规则包 │──┤
                        └──────────────────────┘  │
                                                  │
                        ┌──────────────────────┐  │
                        │ 中文上下文增强器     │──┘
                        └──────────────────────┘
```

系统分成四个可独立版本化的产物：

1. Qwen 模型与 tokenizer；
2. 校准器和类别阈值；
3. Presidio 融合逻辑；
4. 通用、行业、客户/场景规则包。

不能用重新训练模型来替代一个简单的客户规则变更。

---

## 5. 模型选择与架构方案

### 5.1 基座与教师模型选择

#### 默认学生模型：Qwen3-0.6B-Base

理由：

- Qwen3 Base 系列采用 Apache-2.0 许可，并覆盖大规模多语言预训练语料。[R1][R22][R23]
- 参数规模适合在 Presidio 服务中常驻、动态批处理和后续量化。
- Hugging Face 已提供 `Qwen3ForTokenClassification`，便于复用配置、分类头和 Hub 生态；本项目在其基础上实现双向注意力版本。[R2]
- 0.6B 足够小，适合作为首个公开、可复现、可被社区实际部署的模型。

建议同时训练 `Qwen3-1.7B-Base` 作为质量学生。它不再承担正式教师角色，而是用于回答一个更有产品价值的问题：在相同数据和 Presidio 融合下，1.7B 是否提供足以覆盖额外延迟和显存的收益。若答案为是，则在 Hugging Face 发布 0.6B 与 1.7B 两个 tier。

#### 判别式教师阶梯

| 层级 | 推荐模型 | 作用 | 是否默认 |
|---|---|---|---|
| T-small | `Qwen3-4B-Base` | 低成本消融、验证蒸馏流水线 | 否 |
| T-main | `Qwen3-8B-Base` | Full-Attention Token Classification 主教师 | **是** |
| T-upper | `Qwen3-14B-Base` | Dense 性能上界、长尾和困难边界教师 | 条件启用 |
| T-moe | `Qwen3-30B-A3B-Base` | MoE 研究上界；验证总容量与专家路由是否带来额外收益 | 仅研究 A/B |

Qwen3-8B-Base 与 14B-Base 都是预训练 Base 模型，适合挂 Token Classification 头并进行双向改造；其模型仓库采用 Apache-2.0 许可。[R22][R23] 官方还提供 `Qwen3-30B-A3B-Base`，Transformers 提供 `Qwen3MoeForTokenClassification`；但该模型为 30.5B 总参数、约 3.3B 激活参数的 MoE，不能按“30B dense 教师”理解。[R45][R46][R47]

**tokenizer 兼容性必须实测，不能只凭模型家族名称推断**。为每个 checkpoint 记录：

- `tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`、added tokens 的 SHA256；
- vocab size 和 token-id 映射；
- 同一中文 Unicode 压力集的 `input_ids` 与 `offset_mapping`；
- normalization、BOS/EOS/padding 行为；
- chunk ownership 和 special-token 插入位置。

只有以上全部一致时，教师与学生才能直接做 token-logit KL。否则统一回到字符 span 蒸馏，或在字符网格上构造软标签后再投影到学生 token。

选择规则：

- 先训练 4B 或 8B 教师；正式实验默认以 8B 为主教师。
- 教师在冻结金标集上的 Strict span Micro-F1 应明显高于未蒸馏学生，且 Tier-0/Tier-1 Recall 和最差域都更好，才进入蒸馏。
- 14B 相对 8B 必须在至少 3 个种子上有稳定收益，并通过 paired bootstrap；最终还要看**蒸馏后学生**是否提升。
- 建议停止条件：14B 方案相对 8B 方案对学生 Strict span F1 的平均提升小于 0.2–0.3 个百分点，或没有改善 Tier-0 Recall/域外泛化时，不再扩容。
- 只有 14B 已有稳定学生增益、仍存在明确长尾缺口且训练预算允许时，才运行 30B-A3B MoE；需同时监控 router auxiliary loss、专家利用率、吞吐、峰值显存和蒸馏后学生收益。
- 30B-A3B 相对 14B 的 go/no-go 仍以学生端到端指标与单位增益成本为准；若只提升教师自身指标而不提升学生或 Presidio 系统，则不保留。
- 不按参数量自动判定教师质量；边界准确性、校准、困难负样本误报和域外泛化才是准入条件。

#### 生成式标注与仲裁教师

生成式教师与判别教师职责不同。推荐优先使用自托管、开放权重模型：

| 候选 | 推荐角色 | 说明 |
|---|---|---|
| `Qwen/Qwen3.6-27B` | **默认高质量标注/仲裁教师** | 27B dense、Apache-2.0；以 text-only 服务运行，适合少量高价值困难样本与二次审校。[R38][R40] |
| `Qwen/Qwen3.6-35B-A3B` | 高吞吐候选 | 35B 总参数、约 3B 激活；是否优于 27B 必须以中文 PII 标注准确率和单位合格样本成本实测。[R39][R40] |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | 兼容性回退 | 推理生态成熟，Apache-2.0；用于部署栈尚未支持 Qwen3.6 时。[R24] |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | 极少量研究上界 | 只用于高价值争议样本或 taxonomy 压力测试，不是常规数据生产路径。[R25] |

Qwen3.6 官方将 27B 描述为带视觉编码器的后训练因果模型，并采用 Gated DeltaNet 与 Gated Attention 的混合布局；官方也提供 text-only serving 选项。它不应直接替代 Qwen3 Base 判别教师，而应通过严格 JSON 协议输出字符级 span。[R38][R40]

生成式教师的排名标准不是通用榜单，而是：

```text
有效 JSON 率
× quote/offset 精确率
× span Strict F1
× 长尾召回
× hard-negative precision
÷ 单条合格标注成本
```

默认采用 27B，并把 35B-A3B 作为吞吐/成本 A/B；若 35B-A3B 的单位合格样本成本更低且质量不退化，再切换。

#### 为什么不继续把 1.7B 作为最终教师

1.7B 可用于快速打通训练管线，但它与 0.6B 的容量差距有限，未必产生稳定的 soft-target 增益。正式方案把 1.7B 定位为“质量学生”，把 4B 定位为低成本教师消融，将 8B 设为默认判别教师、14B 设为上界；最终是否保留均由中文 PII 冻结评测决定。

### 5.2 为什么不能只挂分类头

Token Classification 需要同时利用实体左右两侧上下文。标准 decoder-only Qwen 使用因果注意力，位置 `i` 的隐藏状态看不到 `i` 右侧 token。2026 年的 NER 研究表明，把因果注意力替换为 full attention，再做标签监督微调，可以明显改善 decoder-only 模型的 NER 表现；该研究包含 Qwen2.5 和 LoRA/全量微调对照。[R3]

因此必须至少比较：

- 因果注意力基线；
- Full Attention；
- JPT/Sequence Repetition。

### 5.3 三条训练路线

#### 路线 A：Full Attention，推荐主线

把所有非 padding token 之间的注意力设为双向，保留 RoPE、MLP、RMSNorm 和分类头。

优点：

- 原序列长度不翻倍；
- 与 Token Classification 任务形式一致；
- 当前 NER 研究证据最直接。[R3]

风险：

- 需要维护 Qwen3 模型适配层；
- Transformers 内部 mask 实现会变化；
- 自定义模型的 ONNX/FlashAttention 导出需要额外验证。

#### 路线 B：JPT / Sequence Repetition，架构兼容备选

输入概念上变为：

```text
[BOS] 原文 [DELIMITER] 原文 [EOS]
```

不要假设 Qwen tokenizer 原生存在 `[SEP]`。首选复用已存在且不会出现在普通文本中的控制 token；若新增 `<|pii_sep|>`，必须同步扩展 tokenizer、resize embeddings，并把它写入模型发布契约。只对第二份原文计算 Token Classification loss。第二份中的每个 token 都能通过第一份原文看到完整句子。JPT 和 Sequence Repetition 的 2026 研究均表明该思路可为 decoder-only 模型提供有效双向上下文，且不必修改注意力结构。[R4][R5]

优点：

- 不修改基础模型 attention mask；
- 便于验证 Full Attention 实现是否正确；
- 对混合/线性注意力模型也较友好。

缺点：

- 序列长度约翻倍；
- 全注意力计算下成本可能接近四倍；
- 切窗更复杂；
- 标准 HF Token Classification pipeline 需要自定义预处理和只解码第二份输入。

#### 路线 C：因果注意力基线

直接使用 `Qwen3ForTokenClassification`。只作为最低基线和部署兜底，不应在没有和 A/B 路线比较时直接上线。

### 5.4 Full Attention 实现约束

实现一个 `Qwen3BiForTokenClassification`，要求：

- 继承或封装 `Qwen3ForTokenClassification`；
- 关闭 `use_cache`；
- 将内部 causal mask 替换为仅屏蔽 padding 的 4D additive mask；
- attention kernel 明确以 `is_causal=False` 运行；
- 首先用 `eager` 或 SDPA 验证正确性，再启用优化 kernel；
- 锁定 Transformers 版本和 commit；
- 保存自定义架构版本号，例如 `qwen3_bi_token_cls_v1`。

概念性 mask（非 packed batch）：

```python
# attention_mask: 1=有效 token，0=padding，shape=[B, L]
# 只屏蔽无效 key；padding query 的输出稍后丢弃。不要把 padding query
# 的整行都设为 -inf，否则部分 attention kernel 可能产生 NaN。
valid_key = attention_mask[:, None, None, :].bool()   # [B, 1, 1, L]
full_mask = torch.zeros(
    (attention_mask.size(0), 1, attention_mask.size(1), attention_mask.size(1)),
    dtype=dtype,
    device=attention_mask.device,
)
full_mask = full_mask.masked_fill(~valid_key, torch.finfo(dtype).min)
```

若启用 sequence packing，必须额外构造 block-diagonal document mask，禁止不同文档互相注意。当前 Qwen3 模型会在内部创建 causal mask。**不能假设设置一个配置字段或传入普通 2D `attention_mask` 就自动取消因果性**；必须覆盖内部 mask 构造路径并做单元测试。

#### 必做正确性测试

1. 修改某个未来 token，较早位置的 logits 在 Full Attention 模式下必须发生变化；因果基线下不应变化。
2. 改变 padding token 内容不能影响有效 token logits。
3. `save_pretrained` 后重新加载，固定输入的 logits 保持一致。
4. batch 和逐条推理的 span 完全一致。
5. BF16、FP32、SDPA/eager 的结果在允许误差内一致。
6. 梯度可以从早期 token 的分类 loss 传播到未来 token embedding。

### 5.5 双向领域适配

Full Attention 改变了预训练时的可见上下文分布。建议在监督微调前增加一个可选适配阶段：

1. 收集安全处理后的简体中文领域文本；
2. 将真实 PII 替换为保持格式的合成值；
3. 添加专用 `<|pii_mask|>` token；
4. 随机 mask 15%–20% token，并增加连续 span mask；
5. 用双向上下文预测被 mask token；
6. 仅训练 LoRA/部分层，保留基础能力。

该阶段受 LLM2Vec 的“启用双向注意力 + masked next token prediction”思想启发。[R6]

建议实验规模：

- 快速验证：5,000 万–1 亿 token；
- 生产候选：2 亿–10 亿 token，按领域覆盖决定；
- 不在该阶段直接使用未经授权或未替换的客户真实 PII。

### 5.6 分类头和损失

#### 主输出

- BIO 标签；
- `O` + 每类 `B-`/`I-`；
- special token、padding、非训练副本均使用 `-100`。

选择 BIO 而非 BIOES 的原因：

- Hugging Face Token Classification pipeline 和 Presidio 的直接 recognizer 对 B/I 聚合最稳定；
- 减少自定义解码要求；
- 单 token 实体用 `B-X` 表示即可。

#### 推荐损失

```text
L = L_token + λ_boundary · L_boundary
            + λ_distill · L_distill
            + λ_consistency · L_consistency
```

- `L_token`：带上限的类别加权 Cross Entropy，默认主损失；
- `L_boundary`：可选的实体 start/end 二分类辅助头；
- `L_distill`：8B/14B 判别式教师到 0.6B 的 token KL、span 监督和可选中间层对齐；
- `L_consistency`：原文本与可对齐噪声增强版本之间的一致性损失。

不建议主线一开始加 CRF：

- 它会增加导出和 HF pipeline 兼容成本；
- 首先用 BIO 合法转移约束的 Viterbi 后处理进行消融；
- 如果 CRF 的严格 span F1 提升不足 0.3 个百分点，不保留。

### 5.7 Token 边界和字符 span

所有数据以字符 span 为真值，token 标签只是训练视图。

规范：

- Python 字符索引，左闭右开；
- 保存 `text[start:end] == entity_text` 校验；
- tokenizer 使用 `return_offsets_mapping=True`；
- 不使用 `word_ids()` 假设中文有空格分词；
- 不在推理前做不可逆 NFKC/空格归一化；如规则使用归一化文本，必须保存原文偏移映射。

#### Token 边界不可对齐问题

Qwen BPE token 可能跨越实体边界。数据构建阶段必须统计：

```text
unalignable_boundary_rate =
  实体起止边界不落在 tokenizer offset 边界上的实体数 / 总实体数
```

处理策略：

- 基线：给所有与实体有交集的 token 标 BIO，解码时使用 token offset 扩展；
- 原始金标边界不改写；
- 若不可对齐率超过 0.5%，或造成严格 span F1 明显损失，启用边界细化头；
- 边界细化头预测首尾 token 内的字符偏移，再映射回原文；
- Presidio 最终结果始终输出精确字符 offset。

### 5.8 长文本与切窗

不要直接沿用固定字符切窗作为最终方案。实现 tokenizer-aware chunker：

1. 优先按段落、句号、换行、JSON 字段和表格行切分；
2. 超长片段再按 token 切窗；
3. 初始最大长度 512 token，长文候选 1,024 token；
4. stride 取窗口的 20%–25%；
5. 每个窗口只对中心 ownership 区域提交结果；
6. 跨窗重复 span 进行字符级去重和合并；
7. 训练时避免实体被硬切，或对边界样本设置忽略标签；
8. JPT 路线必须将“原文长度 × 2 + 特殊 token”计入上限。

### 5.9 多教师与蒸馏设计

#### 5.9.1 教师角色分工

不要把所有教师混成一种输出。建议明确四类教师：

| 教师 | 输出 | 可用于的监督 | 主要优势 |
|---|---|---|---|
| `T_qwen_cls` | 每个输入 token 的 label logits + span | token KL、hard BIO、span、feature | 与学生 tokenizer 一致，监督最完整 |
| `T_qwen_gen` | 字符级 JSON span、候选标签、不确定性 | hard/soft span、样本生成、仲裁 | 开放类别理解和困难样本解释 |
| `T_ernie` | 字符 span | span 级蒸馏 | 中文通用人名/地名已有经验 |
| `T_rule` | span + validator 状态 | 高精硬标签、拒绝条件 | 身份证、银行卡、邮箱等确定性强 |

`T_qwen_cls` 默认是 Qwen3-8B Full-Attention Token Classifier；`T_qwen_gen` 默认是自托管 Qwen3.6-27B；Qwen3.6-35B-A3B 是高吞吐候选，Qwen3-30B-A3B-Instruct-2507 是兼容性回退。

#### 5.9.2 判别式教师蒸馏损失

当教师与学生的 tokenizer 文件 hash、词表/id 映射、special tokens、offset 行为、标签顺序和切窗完全一致时，才使用 token-logit KL：

```text
L_total = L_gold
        + λ_pseudo · w(x) · L_hard_pseudo
        + λ_kl · T² · KL(p_teacher^T || p_student^T)
        + λ_span · L_span
        + λ_feat · L_feature
        + λ_cons · L_consistency
```

建议起点：

- temperature `T=2–4`；
- `λ_kl=0.5–1.0`；
- `λ_pseudo=0.25–0.75`，随伪标签置信度和来源调整；
- `λ_span=0.2–0.5`；
- `λ_feat=0–0.1`，只有消融证明有效时保留；
- 金标样本始终保留 `L_gold` 权重 1.0；
- 对 `O` 类 KL 可降权，避免教师的海量背景概率淹没稀有 PII。

中间层对齐不要按相同层号硬配。可把 8B/14B 的层均匀映射到学生层，先投影到学生隐藏维度，再用 cosine/MSE；如果收益低于 0.2 F1 或显著增加训练不稳定性，应删除 feature loss。

#### 5.9.3 生成式教师标注协议

生成式教师必须返回可机器验证的 JSON，禁止自由文本作为训练标签：

```json
{
  "schema_version": "pii-zh-v1",
  "entities": [
    {
      "start": 4,
      "end": 7,
      "label": "PERSON_NAME",
      "quote": "张小明",
      "confidence": 0.94,
      "alternatives": []
    }
  ],
  "uncertain_spans": [],
  "document_has_pii": true
}
```

本地 validator 必须逐项检查：

- `0 <= start < end <= len(text)`；
- `text[start:end] == quote`；
- label 在冻结 taxonomy 中；
- span 不重复，冲突按规则处理；
- 格式型 PII 通过对应校验器或被标记为 `context_only`；
- 生成式教师不得“修正”原文、补写输入中不存在的实体值；
- 失败输出全部隔离，不尝试自动猜测 offset。

不要保存或发布教师的长推理过程。只保留结构化 span、候选标签、短理由代码、模型版本、prompt hash、采样参数和审核状态。

#### 5.9.4 不确定性感知与多轮标注

只对高价值样本使用 2–3 次独立采样或多教师投票：

1. 第一次抽取确定 span；
2. 第二次作为批评器检查遗漏、边界和标签；
3. 第三次只处理前两次分歧；
4. 规则、ERNIE、判别式教师和生成式教师统一到字符 span；
5. 高一致性样本进入伪标签池；分歧样本进入人工标注队列。

可借鉴候选标签蒸馏思想：教师不确定时输出多个候选，不强迫单一标签；学生或人工审核再消解。LLM 生成标注存在固有不确定性，不能把单次高置信文本当作真实概率。[R35][R36][R37]

#### 5.9.5 伪标签权重

为每个 span 计算来源分数：

```text
w_span = w_source
       × agreement_factor
       × calibration_factor
       × validator_factor
       × domain_factor
```

建议：

- 金标：`1.0`；
- 规则校验通过且上下文匹配：`0.95–1.0`；
- 8B/14B 判别教师校准后高置信：`0.7–0.9`；
- 多教师一致：额外乘 `1.1`，最终截断到 `1.0`；
- 单个生成式教师：最高 `0.6`；
- API/生成教师边界或标签不一致：`0`，转人工；
- 无实体文档只有在至少两个模型和规则均无命中时才作为高置信 hard negative。

#### 5.9.6 教师准入与停止条件

教师必须满足：

- Strict span F1 明显高于学生基线；
- Tier-0/Tier-1 Recall 不低于金标训练学生；
- `hard_negative_test` 误报不恶化；
- ECE 或风险-覆盖曲线可校准；
- 领域最差组没有 collapse；
- 对 1,000–2,000 条真实中文困难样本进行人工盲审；
- 蒸馏后学生在至少 3 个随机种子上有稳定提升。

若更大教师未达到这些条件，保留较小教师。教师尺寸不是交付指标。

#### 5.9.7 API 教师合规门禁

第三方 API 可以显著提升标注覆盖和冲突仲裁效率，但它同时引入两类相互独立的风险：

1. **隐私/数据处理风险**：供应商是否训练、保留、人工审核、跨境处理输入输出；
2. **输出使用/派生模型风险**：是否允许把输出作为训练目标，以及是否允许公开、商业分发派生权重。

“供应商不会用你的数据训练自己的模型”只回答第 1 类问题，不能推出第 2 类权限。

##### 截至 2026-07-10 的工程门禁矩阵

| 服务来源 | 标准公开条款的关键点 | 对公开 HF 模型的默认结论 |
|---|---|---|
| OpenAI API/Business | 限制针对使用输出开发与 OpenAI 竞争的 AI 模型；分类器例外要求不得向第三方分发或商业提供。PII 分类器是否构成“竞争模型”需要按具体用途和合同判断 | **默认不进入公开训练池**；只有 Order Form 或供应商书面确认明确覆盖该用途、开源权重训练与第三方分发时才例外。[R28] |
| Anthropic API | 公开说明列出 specialized classifiers/information extraction 为可接受用途，但同时要求书面许可，并把“使用输出作为训练目标”列入禁止示例 | 存在文本冲突/解释风险；**必须取得书面确认**，否则只做人工辅助或内部评测。[R29] |
| Gemini API | 禁止开发与服务竞争的模型；免费服务会用内容改进产品并可能人工审核，且不得提交敏感/机密/个人信息；付费服务的数据处理条款更好 | 付费与 DPA 只解决部分隐私问题；公开训练/分发仍需书面确认。免费服务不得用于 PII 语料。[R30] |
| 阿里云百炼 | 隐私页称不会将客户数据用于模型训练；但模型体验输出仅限体验，限制商业化与第三方传播 | 不使用模型体验输出；正式付费推理 API 需审核服务协议、订单和模型提供方条款。[R31][R43][R44] |
| 自托管 Apache-2.0 Qwen | 模型权重与推理在自有环境，来源、prompt 和保留策略可控 | **默认推荐**；仍需审查输入数据许可、模型许可证、输出中真实 PII 与开源数据链。[R38][R39][R40] |

以上是工程保守策略，不替代针对具体合同的法律意见。条款会变更，因此必须保存调用时适用的版本与证据。

##### 进入 `public_release_pool` 的必要条件

将 API 输出用于公开权重训练前，必须由法务对具体账号、产品、地区、合同版本和订单条款做书面确认，至少覆盖：

- 允许把输出作为模型训练目标或伪标签；
- 允许公开分发和商业使用派生权重；
- 允许保留、再处理并发布必要的衍生统计/数据；
- 不受竞争产品/模型限制，或已明确确认 PII Token Classifier 不受该限制；
- 输入数据具备发送给供应商的授权；
- 数据保存期、处理地区、跨境机制、人工审核和删除机制满足要求；
- 是否要求归因、AI 生成标识、删除、审计或 downstream 通知；
- 条款变化后，历史输出适用哪个合同版本。

API 只允许处理纯合成、公开许可文本或已经不可逆去标识化的文本；**不得把需要检测的真实客户 PII 直接发给第三方 API 来“帮忙标注”**。

##### 来源隔离与可撤销性

建立 `teacher_provenance.jsonl`，并确保任何 API 来源都能从训练集和权重配方中单独移除：

```json
{
  "sample_id": "sha256:...",
  "provider": "self_hosted_qwen",
  "model": "Qwen3.6-27B",
  "model_revision": "<commit>",
  "terms_or_license": "Apache-2.0",
  "terms_effective_date": "2026-04",
  "prompt_hash": "sha256:...",
  "generation_config_hash": "sha256:...",
  "input_class": "synthetic_only",
  "output_role": "candidate_span_and_adjudication",
  "quality_tier": "S1_multi_teacher_agreement",
  "allowed_for_public_training": true,
  "legal_review_id": "OSS-PII-TEACHER-001"
}
```

所有 API 派生样本放在独立 shard，并维护“无 API 重新训练”配置。若未来条款、授权或来源审计失败，可以删除对应 shard 并从确定性配置重训公开模型。

标准商业 API 条款不能靠“输出归用户所有”推导出可训练并开源模型。若公开分发授权不明确，设置 `allowed_for_public_training=false`，该来源只能用于内部对照、错误分析和人工审核辅助。[R28][R29][R30][R31]

---

## 6. 数据集方案

### 6.1 统一数据格式

所有来源先转成统一 JSONL：

```json
{
  "doc_id": "sha256-or-uuid",
  "text": "客户张明的联系电话是13812345678。",
  "entities": [
    {
      "start": 2,
      "end": 4,
      "label": "PERSON_NAME",
      "text": "张明",
      "source": "gold",
      "risk_tier": "T1"
    },
    {
      "start": 11,
      "end": 22,
      "label": "PHONE_NUMBER",
      "text": "13812345678",
      "source": "gold",
      "risk_tier": "T1"
    }
  ],
  "language": "zh-Hans-CN",
  "domain": "customer_service",
  "scene": "after_sales",
  "tenant_group": "tenant_hash",
  "conversation_group": "thread_hash",
  "template_group": null,
  "entity_value_groups": ["hashed-normalized-values"],
  "created_at_bucket": "2026-Q2",
  "license": "internal-approved",
  "generator_version": null,
  "split": null
}
```

禁止把 BIO token 序列作为唯一真值存储。

### 6.2 公共数据使用矩阵

| 数据集 | 角色 | 公开信息 | 使用决策 |
|---|---|---|---|
| `ai4privacy/pii-masking-openpii-1.5m` | warm-up | 1,636,375 条、30 种语言、19 类、CC-BY-4.0，包含 `zh-CN`，提供字符 span 和预计算 BIO 标签。[R11] | 只过滤 `language=zh, region=CN`；忽略其 mBERT token 标签，重新对齐；先人工抽检和模板去重 |
| `wan9yu/pii-bench-zh` | 冻结外部测试 | 5,000 正式文本 + 3,000 chat，共 23,206 个实体，8 类，Apache-2.0，100% 合成。[R12] | 严禁训练、阈值调参、蒸馏和模板复用；单独报告 formal/chat |
| `CLUENER2020` | 通用中文 NER warm-up | 10 类，训练集 10,748、验证集 1,343，可提供姓名和地址等通用 NER 监督。[R13] | 仓库页面未给出明确许可证；商业项目在法务确认前不进入正式训练，只可做研究对照 |
| `CyberChangAn/MultiPriv` | 非商业研究参考 | 中英文、多模态，含 1,000 条中文个人记录，CC-BY-NC-SA-4.0。[R14] | 默认排除商业训练；可借鉴标签定义或在隔离研究分支评估 |
| 自建合成集 | 主训练集 | 自有模板、值生成器、噪声和场景 | 记录 generator/template/seed；与测试模板家族隔离 |
| 内部金标集 | 最终训练/验证/测试 | 真实业务分布 | 安全环境标注；客户、时间、线程、实体值分组切分 |

### 6.3 OpenPII 处理流程

1. 过滤 `zh-CN`；
2. 丢弃 placeholder 未替换、语言错配、明显乱码和 span 校验失败样本；
3. 对 2,000 条随机样本做人工质量审计；
4. 按模板/近重复聚类，限制单一模板占比；
5. 进行标签映射：
   - `GIVENNAME` + 相邻 `SURNAME` 合并为 `PERSON_NAME`；
   - `TELEPHONENUM` → `PHONE_NUMBER`；
   - `EMAIL` → `EMAIL_ADDRESS`；
   - `CREDITCARDNUMBER` → `BANK_CARD_NUMBER`；
   - `IDCARDNUM` 先按中国格式校验，再映射 `CN_RESIDENT_ID` 或通用 ID；
   - `STREET/CITY/BUILDINGNUM/ZIPCODE` 在同一地址短语内合并为 `ADDRESS`；
   - `DATE` 只有在出生上下文中才映射 `DATE_OF_BIRTH`，否则映射辅助 `GENERIC_DATE` 或忽略；
6. 用 Qwen tokenizer 重新生成 offsets 和 BIO；
7. 产出数据质量报告和许可清单。

### 6.4 自建合成数据

#### 场景覆盖

至少覆盖：

- 电商注册、支付、退款、物流、售后；
- 银行、证券、保险、信贷；
- 医疗预约、病历、检验、医保；
- 人力资源、招聘、员工档案；
- 教育、学生服务；
- 酒店、机票、签证和出行；
- 政务服务；
- 即时通信、群聊、客服对话；
- 软件日志、异常栈、JSON、SQL、配置文件；
- 合同、邮件、工单、表格和 Markdown。

#### 安全生成方法

推荐使用“占位符生成 → 文本改写 → 合成值替换”的顺序：

```text
请联系 <<PERSON_1>>，手机是 <<PHONE_1>>。
        ↓ 仅对占位符文本做 LLM 改写
麻烦直接找一下 <<PERSON_1>>，电话留的是 <<PHONE_1>>。
        ↓ 用本地生成器替换
麻烦直接找一下 周晓宁，电话留的是 139 5821 7046。
```

好处：

- LLM 不接触真实 PII；
- 不会在改写时篡改身份证号或手机号；
- 替换后可以确定性计算精确 offset；
- 可复现每个样本的模板、seed 和生成器版本。

#### 值生成器

应实现并测试：

- 大陆身份证：地区码、合法日期、MOD 11-2 校验位；
- 银行卡：可配置长度、Luhn 校验，避免生成真实完整账户；
- 手机/座机/分机及带空格、短横线格式；
- 邮箱、IP、MAC、经纬度；
- 护照、驾驶证、车牌常见形式；
- 微信号、QQ、支付宝账号；
- 病历号、工号、学号的行业模板；
- 姓名、少数民族姓名、复姓、英文名和中英混排；
- 省市区街道、农村地址、楼栋单元房间；
- 密码/API key/token 等安全敏感字符串。

每个生成器必须包含：

- 格式有效样本；
- 格式无效但相似样本；
- 变形和分隔样本；
- 负样本；
- 单元测试和版本号。

#### 噪声增强

- 全角/半角和大小写；
- 中文/英文标点；
- 空格、短横线、换行插入；
- 零宽字符和不可见字符；
- OCR 混淆：`0/O`、`1/l/I`、汉字同形；
- ASR 形式：数字汉字化、口语填充词、同音错误；
- emoji；
- 简繁混排；
- 拼音、英文和中文混排；
- Markdown/HTML/JSON 转义；
- 截断、掩码和部分隐藏；
- 多轮对话中的省略和指代。

所有增强必须保留或重建字符偏移映射。

### 6.5 困难负样本

至少 30% 文档不含 PII，且在训练和测试中都要有高相似度负样本：

- 订单号、合同号、快递单号、发票代码；
- 商品编码、版本号、模型号、端口号；
- 普通日期、数量、金额和百分比；
- 企业统一社会信用代码；
- 公司地址和公共客服电话；
- 新闻中的组织、人名和景点；
- 无效身份证、无效银行卡和随机长数字；
- 已脱敏字符串，如 `138****5678`、`张*`；
- 代码、hash、UUID、trace ID；
- “身份证号”上下文后没有真正号码；
- 允许列表中的测试账号和公开服务号。

### 6.6 内部金标数据

#### 建议规模

| 阶段 | 文档数 | span 数 | 用途 |
|---|---:|---:|---|
| 最小可行金标 | ≥ 10,000 | ≥ 30,000 | 初步领域验证 |
| 生产候选 | 50,000–100,000 | ≥ 200,000 | 跨客户、跨场景上线验证 |
| 每个核心高风险类别测试 span | ≥ 500 | — | 稳定置信区间 |
| 每个低频类别测试 span | ≥ 200 | — | 最低可解释性 |
| 无 PII 测试文档 | ≥ 10,000 | 0 | 误报和过度脱敏评估 |

#### 标注流程

1. 在隔离环境抽样；
2. 使用当前 Presidio + ERNIE + 规则做预标注，但保留一部分完全空白预标注样本，测量预标注偏差；
3. 至少 15%–20% 双人独立标注；
4. 冲突由高级标注员裁决；
5. 记录 span 边界一致率、类型一致率和标注员间 span-F1；
6. 每轮更新标注指南和边界示例；
7. 测试集冻结后不得再用于提示、规则调试或阈值选择。

#### 数据保护

- 标注平台部署在受控网络；
- 最小权限和操作审计；
- 数据静态/传输加密；
- 禁止把原始 PII 写入实验追踪、异常日志和第三方 SaaS；
- 导出给训练前优先做一致格式的合成替换；
- checkpoint、梯度和缓存按敏感数据管理；
- 训练结束后执行保留期限和删除策略。

### 6.7 多教师数据质量分层与公开/私有双轨

每条训练样本必须同时有“质量等级”和“公开资格”，两者不能混为一谈。

| 等级 | 定义 | 默认训练权重 | 是否可公开训练 |
|---|---|---:|---|
| G0 | 双标/裁决后的真实金标 | 1.00 | 仅当数据与派生权重授权明确 |
| S0 | 确定性合成 + 校验器完全通过 | 0.90–1.00 | 通常可以，取决于模板和值源许可证 |
| S1 | 至少两个独立教师一致，且 quote/offset/规则验证通过 | 0.70–0.90 | 取决于每个教师来源许可 |
| S2 | 单一教师高置信且验证通过 | ≤0.60 | 默认谨慎，需专项抽检 |
| U | 教师分歧、候选多标签或边界不确定 | 0 | 转人工或 candidate distillation |
| N | 经确认的困难负样本 | 0.50–1.00 | 取决于文本来源许可 |

数据池物理隔离：

```text
data/processed/
├── public_release_pool/       # 可训练并公开派生权重
├── private_enterprise_pool/   # 内部/客户授权，不进入公共权重
├── evaluation_only/           # 外部测试、冻结集，绝不训练
└── quarantined/               # 来源/许可/质量未决
```

任何样本只有同时满足 `quality_gate=true` 和 `public_weight_training_allowed=true` 才能进入开源训练。研究表明，LLM 合成监督和 teacher-student 数据流水线可以有效扩展 NER，但也需要系统化筛选、候选标签和独立验证，而不是把单次生成直接当金标。[R35][R36][R37][R41][R42]

### 6.8 数据切分与泄漏防护

禁止普通随机句子切分。使用 group split：

- 同一客户/tenant；
- 同一会话/thread；
- 同一文档；
- 同一合成模板家族；
- 同一生成 seed 系列；
- 同一归一化实体值 hash；
- 同一来源和时间窗口；
- 近重复文本 MinHash/SimHash 聚类。

建议测试结构：

1. `internal_iid_test`：同领域、不同文档；
2. `tenant_holdout_test`：未见客户；
3. `domain_holdout_test`：未见场景；
4. `time_holdout_test`：较新时间段；
5. `hard_negative_test`；
6. `noise_stress_test`；
7. `long_document_test`；
8. `pii_bench_zh_formal`；
9. `pii_bench_zh_chat`。

---

## 7. 训练方案

### 7.1 分阶段训练

#### 阶段 T0：基线冻结

建立并保存：

- 当前 ERNIE 模型单独结果；
- 当前规则单独结果；
- 当前 ERNIE + 规则端到端结果；
- Presidio anonymizer 最终泄漏和过度遮盖指标。

#### 阶段 T1：公共/合成 warm-up

- `Qwen3-0.6B-Base`；
- 因果注意力和 Full Attention 同时跑；
- 以 OpenPII zh-CN + 自建合成数据训练；
- 不使用内部测试集；
- 目标是验证管线、标签和 offset，而非上线。

#### 阶段 T2：双向领域适配

- Full Attention；
- 使用安全处理后的领域无标注语料做 span-mask/MNTP 风格适配；
- 对比无适配版本；
- 保留 tokenizer 和 embedding 变更记录。

#### 阶段 T3：内部金标监督

建议 curriculum：

1. 清洁公共/合成；
2. 噪声增强和困难负样本；
3. 内部金标混合训练；
4. 最后一轮只用高质量金标、小学习率收敛；
5. 冻结模型后做 calibration，不再更新权重。

#### 阶段 T4：教师训练、多教师标注和蒸馏

- 先训练 `Qwen3-8B-Base` Full-Attention Token Classification 教师；
- 以 `Qwen3-14B-Base` 做判别教师上界，只有相对 8B 使蒸馏学生稳定提升时保留；
- `Qwen3-4B-Base` 用于低成本教师消融；`Qwen3-1.7B-Base` 作为质量学生而非正式教师；
- 教师、ERNIE 和高可信规则统一输出字符 span；
- 自托管 `Qwen3.6-27B` 作为默认生成式标注/仲裁教师；并与 `Qwen3.6-35B-A3B` 比较“单位合格标注成本”；
- 部署栈暂不支持 Qwen3.6 时，回退到 `Qwen3-30B-A3B-Instruct-2507`；
- 生成式教师只处理合成、公开或已不可逆去标识化文本，并通过 JSON Schema/约束解码、quote-offset 校验、第二遍 critic 和规则验证；
- 对无标注文本产生带来源、置信度、agreement、validator 状态和公开资格的伪标签；
- 规则只有在格式、校验位和必要上下文均通过时才能作为硬标签；
- 8B/14B 与学生 tokenizer **经过完整 hash/offset 一致性验证**后，才能做 token-logit KL；否则只做字符 span 蒸馏；
- ERNIE、生成式教师和外部 API 一律只做字符 span 级监督；
- API 教师未经开源训练与公开分发书面授权，不进入 `public_release_pool`；
- 分别蒸馏 0.6B 和 1.7B 学生，最终按端到端效果/成本发布一个或两个检查点；
- 伪标签数据不进入验证/测试，所有来源写入 `teacher_provenance.jsonl`。

### 7.2 推荐超参数起点

#### LoRA 快速迭代

```yaml
base_model: Qwen/Qwen3-0.6B-Base
attention_mode: full
max_length: 512
precision: bf16
use_cache: false
gradient_checkpointing: true
lora:
  rank: 32
  alpha: 64
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
optimizer: adamw
learning_rate_lora: 1.0e-4
learning_rate_head: 5.0e-4
weight_decay: 0.01
warmup_ratio: 0.05
scheduler: cosine
epochs: 3-6
effective_batch_size: 64-256
head_dropout: 0.1
gradient_clip_norm: 1.0
```

#### 全量微调候选

- backbone learning rate：`1e-5` 到 `3e-5` 搜索；
- classifier/head 可使用 5–10 倍学习率；
- 最少运行 3 个随机种子；
- 依据风险加权验证指标早停；
- 0.6B 最终候选必须和 LoRA 比较，全量微调没有稳定提升时不保留。

### 7.3 类别不平衡

主线优先：

- 按文档和稀有类别过采样；
- 类别权重使用 `sqrt(N / N_c)`，上限建议 5；
- 每个 batch 保证有一定比例实体样本和无实体样本；
- 对高频 `O` 不做简单 token 丢弃，避免破坏上下文。

实验分支：

- focal loss，`gamma=1–2`；
- class-balanced loss；
- 辅助混淆标签。

选择标准是严格字符 span 和 PII-free 误报，不是 token accuracy。

### 7.4 增强一致性

对可以确定性对齐的增强，加入一致性训练：

- 全半角；
- 可选空格/短横线；
- 标点替换；
- 大小写；
- emoji 和换行。

对原文和增强文本的对应实体 token 计算对称 KL 或 span 一致性。无法可靠对齐的 LLM 改写不使用 token-level consistency，只作为独立标注样本。

### 7.5 早停指标

建议使用：

```text
score = 0.45 × Tier0_F2
      + 0.25 × Tier1_F2
      + 0.15 × Strict_Macro_F1
      + 0.10 × PII_Free_Precision
      + 0.05 × Calibration_Score
```

具体权重由业务风险调整。不要按训练 loss 或 token accuracy 早停。

---

## 8. 训练实验矩阵

| 实验 | 模型 | 注意力/输入 | 训练方式 | 目的 |
|---|---|---|---|---|
| E0 | 现网 ERNIE | 双向 encoder | 现网 | 系统基线 |
| E1 | Qwen3-0.6B | causal | LoRA | 最低 Qwen 基线 |
| E2 | Qwen3-0.6B | Full Attention | LoRA | 主方法快速验证 |
| E3 | Qwen3-0.6B | Full Attention | 全量微调 | 轻量学生质量上界 |
| E4 | Qwen3-0.6B | JPT ×2 | LoRA | 无改 mask 备选 |
| E5 | Qwen3-0.6B | Full Attention + 双向领域适配 | LoRA/全量 | 验证适配收益 |
| E6a | Qwen3-4B | Full Attention | LoRA/全量 | 低成本教师消融 |
| E6b | Qwen3-8B | Full Attention | LoRA/全量 | 默认判别式教师 |
| E6c | Qwen3-14B | Full Attention | LoRA/全量 | Dense 判别教师性能上界 |
| E6d | Qwen3-30B-A3B-Base | Full Attention + MoE | LoRA/分阶段解冻 | MoE 判别教师研究上界 |
| E7a | Qwen3-0.6B | Full Attention | 8B hard-span 蒸馏 | 验证硬伪标签 |
| E7b | Qwen3-0.6B | Full Attention | 8B token-KL + span | 默认轻量蒸馏方案 |
| E7c | Qwen3-0.6B | Full Attention | 14B token-KL + span | 验证更大 dense 判别教师价值 |
| E7d | Qwen3-1.7B | Full Attention | 8B/14B 蒸馏 | 质量学生候选 |
| E7e | 最优学生 | Full Attention | 30B-A3B MoE char-span / token-KL 蒸馏 | 仅在兼容性审计通过后验证 MoE 增益 |
| E8a | 最优学生 | Full Attention | 自托管 Qwen3.6-27B 生成标注 | 默认生成教师消融 |
| E8b | 最优学生 | Full Attention | 自托管 Qwen3.6-35B-A3B 生成标注 | 吞吐/成本对照 |
| E8c | 最优学生 | Full Attention | Qwen3-30B-A3B-2507 生成标注 | 兼容性对照 |
| E9 | 最优学生 | Full Attention | 边界辅助头 | 修复 tokenizer 边界 |
| E10 | 最优学生 | Full Attention | 显式混淆标签 | 降低误报 |
| E11 | 最优学生 | Full Attention | 规则/ERNIE/多教师融合 | 端到端最终系统 |
| E12 | 最优学生 | Full Attention | 第三方 API 教师数据对照 | 仅在合规门禁通过后运行 |
| E13 | 最优学生 | Full Attention | 去除所有 API 派生数据重训 | 验证可撤销性与来源依赖 |

教师比较必须控制：

- 相同训练数据、标签版本、切窗和评测集；
- 相同或经预算归一化的训练 token 数；
- 至少 3 个随机种子；
- paired bootstrap 置信区间；
- 单独报告教师自身、蒸馏后学生和端到端 Presidio 的收益；
- 报告额外训练成本、GPU 小时、合格伪标签率和 API/标注成本；
- 生成教师以字符 span 质量和单位合格样本成本比较，不以通用聊天榜单替代；
- 14B 的 go/no-go 以蒸馏后学生对 8B 方案的增益为准；30B-A3B MoE 只有在 14B 已通过后才进入，并额外报告路由稳定性与单位学生增益成本。

每个实验必须保存：

- 数据 manifest hash；
- 代码 commit；
- Transformers/PyTorch/Presidio 版本；
- tokenizer 文件 hash 与 offset compatibility report；
- attention mode；
- label schema version；
- teacher/model revision 与 prompt hash；
- source/terms snapshot；
- seed；
- 模型、端到端、性能、校准和成本报告。

---

## 9. 评测方案

### 9.1 模型级指标

必须以字符 span 为主：

- Strict span Micro Precision/Recall/F1；
- Strict span Macro F1；
- 每类别 P/R/F1/F2；
- Exact boundary accuracy；
- Relaxed overlap F1，IoU ≥ 0.5，仅作辅助；
- 字符级 Precision/Recall/F1；
- 混淆矩阵；
- 每 10,000 个实体的 FN 数；
- 每 10,000 字符/文档的 FP 数；
- 无 PII 文档误报率；
- calibration：ECE、Brier score、可靠性曲线。

`seqeval`/token F1 只作为训练诊断，不能作为上线主指标。

### 9.2 端到端 Presidio 指标

比较以下组合：

1. Rules only；
2. ERNIE only；
3. Qwen only；
4. Rules + ERNIE，当前系统；
5. Rules + Qwen；
6. Rules + ERNIE + Qwen；
7. Rules + Qwen + 中文上下文增强；
8. 最终客户场景配置。

端到端指标：

- 金标实体被任何最终结果覆盖的 Recall；
- 金标 PII 字符最终仍未遮盖的比例；
- 非 PII 字符被遮盖的比例；
- 一篇文档内所有 T0/T1 PII 均被处理的 safe-document rate；
- 结果类型错误率；
- 过长/过短边界率；
- 每个 recognizer 的贡献、冲突和独有命中数；
- 按客户、场景、文本长度和噪声类型分层结果。

### 9.3 鲁棒性测试

建立固定压力测试集：

- 电话/身份证插入空格、短横线、换行；
- 全角数字；
- 数字转汉字；
- OCR/ASR 错误；
- 中英文混排；
- 简繁混排；
- emoji；
- JSON/HTML/Markdown/表格；
- 超长文本和实体跨窗口；
- 零宽字符；
- 截断和部分脱敏；
- 多个实体紧邻；
- 同一字符串在不同上下文中分别是 PII 和非 PII；
- 客户新增规则和允许列表。

### 9.4 性能评测

至少测试：

- 序列长度 128/512/1,024；
- batch 1/8/32；
- BF16、可选 INT8/权重量化；
- p50/p95/p99 延迟；
- tokens/s 和 documents/s；
- GPU 显存、CPU 内存；
- 模型首次加载时间；
- Full Attention vs JPT；
- 0.6B 学生 vs 8B/14B 教师；
- 蒸馏前后的 0.6B；
- Presidio 完整请求，而非只测模型 forward。

量化后必须重新跑每类别召回和端到端泄漏指标；不能只比较平均 F1。

### 9.5 阈值和校准

流程：

1. 在验证集上做 token logit temperature scaling；
2. 聚合为 span；
3. 对 span score 做类别级 calibration；
4. 按风险成本选择每类阈值；
5. 测试集只做一次最终评估。

高风险类别以 Recall/F2 和 leakage 为主；姓名/地址等以 Recall 与过度脱敏共同决定。

推荐保存：

```json
{
  "model_version": "qwen3-pii-zh-0.6b-v1",
  "calibration_version": "cal-v3",
  "global_temperature": 1.37,
  "entity_thresholds": {
    "CN_RESIDENT_ID": 0.35,
    "BANK_CARD_NUMBER": 0.40,
    "PERSON_NAME": 0.72,
    "ADDRESS": 0.68
  }
}
```

数值仅为配置示例，必须由验证集得到。

---

## 10. Presidio 集成方案

### 10.1 为什么采用并行 recognizer

Presidio 支持多个 recognizer 以不同机制共同检测 PII，也支持把 Hugging Face Token Classification 模型作为 NLP engine 或额外 recognizer。[R7][R8][R19]

当前系统已经使用 ERNIE 替换 NER，因此最稳妥的迁移是：

- 保持现有 ERNIE 和规则；
- 新增 Qwen recognizer；
- 先并行记录结果；
- 在融合层逐实体启用；
- 最后再决定是否移除 ERNIE 的重叠实体。

### 10.2 Presidio 版本

开发和回归基线建议：

```text
presidio-analyzer >= 2.2.362, < 2.3
```

在调研时最新稳定发布为 2.2.363；2.2.362 引入了直接运行 Hugging Face pipeline、无需 spaCy 对齐的 `HuggingFaceNerRecognizer`。[R9][R10]

### 10.3 标准 HF 模型的最小接入

下面的代码应添加到**现有、已经加载中文 ERNIE NLP engine 的 `AnalyzerEngine`**。不要为了接入 Qwen 重新实例化默认 `AnalyzerEngine()`，否则可能丢失当前中文 NLP engine、已有 recognizer 和客户规则。

该最小路径适用于能被标准 Hugging Face `pipeline("token-classification")` 直接加载的模型。主线自定义 Full-Attention 模型若不能被 `AutoModelForTokenClassification` 无条件加载，应使用 10.4 的自定义 recognizer，而不是依赖 `trust_remote_code` 隐式执行仓库代码。

```python
from presidio_analyzer.predefined_recognizers.ner import (
    HuggingFaceNerRecognizer,
)

MODEL_PATH = "/models/qwen3-pii-zh"

qwen_recognizer = HuggingFaceNerRecognizer(
    name="Qwen3PiiZhRecognizer",
    model_name=MODEL_PATH,
    tokenizer_name=MODEL_PATH,
    supported_language="zh",
    supported_entities=[
        "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CN_ADDRESS",
        "CN_ID_CARD", "PASSPORT_NUMBER", "BANK_CARD_NUMBER",
        "WECHAT_ID", "IP_ADDRESS", "SECRET",
    ],
    label_mapping={
        # HuggingFaceNerRecognizer 会先剥离 B-/I- 前缀，再查映射。
        "PERSON_NAME": "PERSON",
        "PHONE_NUMBER": "PHONE_NUMBER",
        "EMAIL_ADDRESS": "EMAIL_ADDRESS",
        "ADDRESS": "CN_ADDRESS",
        "CN_RESIDENT_ID": "CN_ID_CARD",
        "PASSPORT_NUMBER": "PASSPORT_NUMBER",
        "BANK_CARD_NUMBER": "BANK_CARD_NUMBER",
        "WECHAT_ID": "WECHAT_ID",
        "IP_ADDRESS": "IP_ADDRESS",
        "SECRET": "SECRET",
    },
    threshold=0.20,                 # 召回入口阈值；最终按实体校准和融合
    aggregation_strategy="simple",
    chunk_size=800,                 # 当前类按字符切窗，不是 token 数
    chunk_overlap=160,
    device="cuda:0",
    label_prefixes=["B-", "I-"],
)

# existing_analyzer 是当前已配置 ERNIE、规则及 supported_languages=["zh"]
# 的 AnalyzerEngine 实例。
existing_analyzer.registry.add_recognizer(qwen_recognizer)
```

Presidio 当前实现支持模型路径、label mapping、全局 threshold、`simple/first/average/max` 聚合、字符切窗和 GPU 设备选择。[R9] 生产主线仍应将字符切窗替换成 tokenizer-aware chunker，并确保 recognizer 的 label mapping 与模型 `id2label` 完全一致。

### 10.4 推荐的自定义 `QwenPiiRecognizer`

主线 Full Attention 模型还需要：

- 自定义模型类加载；
- tokenizer-aware 切窗；
- BIO 合法转移约束；
- token 内字符边界细化；
- 类别阈值；
- 校准器；
- 批量推理；
- 更完整的 recognition metadata。

因此建议派生 `HuggingFaceNerRecognizer`，覆盖 `load()` 和 `_predict_chunk()`；或直接继承 `LocalRecognizer`。

接口合同：

```python
class QwenPiiRecognizer(LocalRecognizer):
    def load(self) -> None:
        """加载 tokenizer、Qwen3BiForTokenClassification、校准器和阈值。"""

    def analyze(self, text, entities, nlp_artifacts=None):
        """返回字符级 RecognizerResult；不依赖 spaCy token 对齐。"""

    def analyze_batch(self, texts, entities):
        """服务层批量接口，保证与逐条结果一致。"""
```

每个 `RecognizerResult` 应包含：

- entity type；
- start/end；
- calibrated score；
- 原始 score；
- model version；
- attention mode；
- window id；
- rule/model source；
- boundary refinement 是否发生。

### 10.5 Presidio TransformersNlpEngine 备选

如果未来只保留一个 NER 模型，可使用官方 `TransformersNlpEngine`，其 YAML 支持模型路径、`aggregation_strategy`、`stride`、`alignment_mode` 和 model-to-Presidio mapping。[R7]

当前阶段不推荐直接替换 ERNIE NLP engine，因为：

- 并行 recognizer 更容易 shadow 和回滚；
- Qwen 可以绕过 spaCy 中文 token 对齐；
- 可以按实体逐步启用；
- ERNIE 的通用中文 NER 能力仍可作为补充。

### 10.6 结果融合

#### 初始确定性融合

1. **校验通过的规则**：
   - 身份证 MOD 11-2、银行卡 Luhn 等，设为高置信；
   - 对格式错误结果抑制，而不是仅因正则匹配就高分。
2. **Qwen**：
   - 使用类别校准后的阈值；
   - 对非标准分隔、上下文姓名/地址和账号承担主检。
3. **ERNIE**：
   - PERSON/LOCATION/ORG 辅助；
   - 不直接把所有 LOCATION/ORG 映射为需要脱敏的 PII。
4. **客户规则**：
   - 可覆盖默认阈值、实体类型和动作；
   - 校验规则优先级可配置。

#### 重叠处理

- 同 span、同类型：保留最高校准分并记录来源一致性；
- 同 span、不同类型：具体类型优先于泛化类型；
- 规则识别到身份证而模型识别为通用 ID：保留身份证；
- 宽 span 包含具体高风险 span：先保护具体 span，再按策略决定是否保留宽 span；
- 多结果给同一 anonymizer 动作时先做区间归并，避免重复替换破坏 offset。

#### 学习式融合升级

数据足够后，用验证集训练可解释的逻辑回归 stacking：

特征包括：

- Qwen calibrated score；
- ERNIE score；
- regex score；
- checksum 是否有效；
- 正/负上下文；
- recognizer 一致性；
- 实体长度和格式；
- scene/domain；
- 是否在允许/拒绝列表。

不得用测试集训练融合器。高风险 checksum-valid 规则应保留硬约束，不完全交给学习器。

### 10.7 中文上下文增强器

Presidio 默认 lemma context 主要依赖 NLP token/lemma，且现有设计中的 recognizer context 通常是扁平词表；中文短语和“每类不同上下文”更适合自定义实现。[R19][R20]

实现 `ChinesePhraseContextEnhancer`：

- 按字符窗口和字段边界工作；
- Aho-Corasick/trie 或预编译短语匹配；
- 每个实体独立正/负词表；
- 支持 1–20 字符短语；
- 支持 JSON key、表头、冒号前字段名；
- 正上下文增分，负上下文降分；
- 不改变规则校验失败结果；
- 所有增减分写入 decision process。

示例：

```yaml
positive_context:
  CN_ID_CARD: [身份证, 身份证号, 证件号码, 公民身份号码]
  BANK_CARD_NUMBER: [银行卡, 卡号, 收款卡, 开户账号]
  ADDRESS: [收货地址, 家庭住址, 常住地址, 居住地]
  MEDICAL_RECORD_NUMBER: [病历号, 就诊号, 住院号]

negative_context:
  CN_ID_CARD: [订单号, 合同号, 流水号, 商品编号]
  BANK_CARD_NUMBER: [快递单号, 发票号码]
  PERSON_NAME: [公司名称, 产品名称, 游戏角色]
```

### 10.8 通用和客户规则包

#### `cn_common` 规则包

- 大陆居民身份证：长度、地区码、日期、MOD 11-2；
- 手机/座机/分机；
- 邮箱；
- 银行卡 Luhn；
- IP、MAC、经纬度；
- 护照、驾驶证、车牌常见形式；
- 邮编；
- 常见 secret/key/token；
- 已脱敏值识别和策略。

#### 行业规则包

- `finance`：账户号、客户号、贷款号、证券账号；
- `medical`：病历号、就诊号、医保编号；
- `hr`：工号、候选人编号；
- `education`：学号、准考证号；
- `log_security`：API key、JWT、private key、cookie/session。

#### 客户规则包

每个客户/场景配置：

- 自定义正则和校验函数；
- 允许列表/拒绝列表；
- 模型阈值；
- 正负上下文；
- recognizer 权重和启停；
- 冲突优先级；
- anonymizer 动作；
- 生效版本和回滚版本。

示例：

```yaml
schema_version: 1
tenant: bank_a
scene: customer_service
policy_version: 2026-07-01

recognizers:
  qwen:
    enabled: true
    model_version: qwen3-pii-zh-0.6b-v1
  ernie:
    enabled: true
    entities: [PERSON, LOCATION]
  rule_packs: [cn_common, finance]

entities:
  CN_ID_CARD:
    enabled: true
    model_threshold: 0.35
    final_threshold: 0.45
    checksum_valid_score: 0.99
    action:
      operator: mask
      keep_last: 4
  PERSON:
    enabled: true
    model_threshold: 0.72
    final_threshold: 0.75
    action:
      operator: replace
      value: "<姓名>"
  BANK_CARD_NUMBER:
    enabled: true
    model_threshold: 0.40
    final_threshold: 0.50
    action:
      operator: mask
      keep_last: 4

context:
  positive:
    CN_ID_CARD: [身份证号, 证件号码]
    BANK_CARD_NUMBER: [银行卡号, 收款卡]
  negative:
    CN_ID_CARD: [订单号, 流水号]

allow_lists:
  PHONE_NUMBER: ["95588", "10086"]

custom_patterns:
  - entity: CUSTOMER_NUMBER
    pattern_id: bank_a_customer_no_v2
    regex: "<由客户提供并评审的表达式>"
    score: 0.70
    context: [客户号, 客编]
```

---

## 11. 代码工程和产物结构

```text
pii-zh/
├── configs/
│   ├── model/
│   ├── train/
│   ├── data/
│   ├── evaluation/
│   └── presidio/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── manifests/
├── src/pii_zh/
│   ├── taxonomy/
│   ├── data/
│   │   ├── converters/
│   │   ├── validators/
│   │   ├── synthetic/
│   │   ├── alignment/
│   │   └── splitting/
│   ├── models/
│   │   ├── qwen3_bi.py
│   │   ├── qwen3_jpt.py
│   │   ├── boundary_head.py
│   │   └── decoding.py
│   ├── training/
│   ├── distillation/
│   │   ├── teacher_runner.py
│   │   ├── pseudo_label.py
│   │   ├── provenance.py
│   │   └── losses.py
│   ├── calibration/
│   ├── evaluation/
│   ├── rules/
│   ├── fusion/
│   └── presidio/
│       ├── qwen_recognizer.py
│       ├── context_enhancer.py
│       └── token_chunker.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── security/
├── model_cards/
├── release/
│   ├── hf_model/
│   ├── hf_dataset/
│   ├── third_party_notices/
│   └── release_checklist.md
├── reports/
├── LICENSE
├── NOTICE
├── THIRD_PARTY_NOTICES.md
└── Makefile
```

建议组件：

- Hydra/OmegaConf 或等价配置系统；
- DVC/lakeFS/对象存储 manifest 做数据版本；
- MLflow 或离线实验追踪，只写指标和 hash，不写原始文本；
- `accelerate`/FSDP 做训练；
- Ruff、mypy、pytest；
- 训练镜像、推理镜像和 SBOM；
- 所有依赖锁版本。

### 11.1 每个模型发布包必须包含

```text
model/
├── README.md                    # Hugging Face model card
├── LICENSE
├── NOTICE
├── THIRD_PARTY_NOTICES.md
├── config.json
├── configuration_qwen3_bi.py
├── modeling_qwen3_bi.py
├── model.safetensors
├── tokenizer.json / tokenizer_config.json
├── special_tokens_map.json
├── taxonomy.yaml
├── id2label.json
├── calibration.json
├── thresholds.yaml
├── training_manifest.json
├── data_provenance.json
├── teacher_provenance.json
├── evaluation_report.json
├── model-index.yml
├── SECURITY.md
└── checksums.txt
```

公开仓库不上传：optimizer state、梯度、训练缓存、真实客户样本、原始 API prompt/response、包含 PII 的错误日志或 pickle 权重。

### 11.2 CI 必测项

- 字符 offset 与 emoji、全角、ASCII、换行；
- span 与 token label 双向转换；
- Full Attention 未来 token 敏感性；
- padding 隔离；
- save/load parity；
- BIO 合法性和聚合；
- 跨窗实体与去重；
- 身份证/银行卡等校验器；
- Presidio label mapping；
- 融合冲突；
- anonymizer 后 offset 正确；
- batch/单条一致；
- 不在日志中出现测试 PII canary；
- 历史客户回归样本不退化。

### 11.3 开源仓库拓扑

建议拆成三个仓库，职责和许可证分别清晰：

```text
Hugging Face Models
├── <org>/zh-pii-qwen3-0.6b-bi
│   ├── 默认高性价比权重/配置/自定义模型代码
│   ├── 模型卡、评测、阈值和 provenance
│   └── 最小 Presidio 使用示例
└── <org>/zh-pii-qwen3-1.7b-bi        # 条件发布
    ├── 质量档权重与同一 taxonomy
    └── 与 0.6B 的效果/延迟对照

GitHub
└── <org>/pii-zh-qwen
    ├── 数据转换与合成器
    ├── Full Attention 模型实现
    ├── 训练/蒸馏/校准/评测代码
    ├── Presidio plugin 和规则示例
    ├── CI、Docker、SBOM 和文档
    └── 可复现实验配置

可选 Hugging Face Dataset
└── <org>/pii-zh-synthetic
    ├── 仅许可兼容的公共/合成数据
    ├── dataset card、字段定义和生成 manifest
    └── 不包含内部金标、客户数据或冻结私有测试集
```

推荐公开模型名：

```text
zh-pii-qwen3-0.6b-bi-v1.0.0
zh-pii-qwen3-1.7b-bi-v1.0.0   # 仅在质量/成本门槛通过时发布
```

版本由三部分组成：模型实现版本、taxonomy 版本、数据/训练 recipe 版本。公开 release tag 必须不可变；后续修复新发版本，不覆盖已有 tag。

### 11.4 许可证和来源策略

目标许可证：

- GitHub 代码：Apache-2.0；
- 模型权重：目标为 Apache-2.0，前提是全部基础模型、训练数据、教师输出和第三方组件经法务确认兼容；
- 自建合成数据：单独选择明确的数据许可证，例如 CC-BY-4.0 或 Apache-2.0；
- 文档：可使用 CC-BY-4.0；
- 客户规则和客户金标：默认不公开，且不自动获得公共模型训练授权。

必须区分“基础模型许可证”“代码许可证”“数据集许可证”“API 服务条款”和“派生权重发布许可”。不要因为 Qwen 基座是 Apache-2.0，就自动推断混合训练后的权重一定可以用 Apache-2.0 发布。

建立机器可读 `source_registry.yaml`：

```yaml
sources:
  - id: qwen3_0_6b_base
    kind: base_model
    license: Apache-2.0
    revision: "<commit>"
    public_weight_training_allowed: true
  - id: openpii_zh_cn
    kind: dataset
    license: "<verified-license>"
    attribution_required: true
    public_weight_training_allowed: "<legal-review>"
  - id: customer_gold_v1
    kind: private_dataset
    license: proprietary
    public_weight_training_allowed: false
  - id: qwen3_6_27b_teacher
    kind: self_hosted_teacher
    model_revision: "<commit>"
    license: Apache-2.0
    input_class: synthetic_or_public_only
    public_weight_training_allowed: true
  - id: external_api_teacher_x
    kind: api_teacher
    terms_snapshot: "<sha256>"
    legal_review_id: "<ticket>"
    public_weight_training_allowed: false
```

CI license gate：只要任一被实际采样的 source 没有明确许可结论，release job 失败。

### 11.5 Hugging Face 自定义架构发布

Full Attention 不是标准 Qwen3 配置，因此首版需要发布自定义模型代码：

- `configuration_qwen3_bi.py`；
- `modeling_qwen3_bi.py`；
- `config.json` 中的 `auto_map`；
- 明确 `architectures=["Qwen3BiForTokenClassification"]`；
- 固定 `transformers` 最低/最高验证版本；
- 在模型卡中说明加载自定义代码会执行远程仓库代码。

示例：

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

revision = "v1.0.0"  # 不使用浮动 main
model_id = "<org>/zh-pii-qwen3-0.6b-bi"

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModelForTokenClassification.from_pretrained(
    model_id,
    revision=revision,
    trust_remote_code=True,
    use_safetensors=True,
)
```

`trust_remote_code=True` 是采用摩擦和供应链风险。缓解措施：

1. 模型仓库中的 Python 代码保持最小化，无网络访问、无动态下载、无 shell 调用；
2. GitHub 与 HF 代码 commit 一一对应；
3. release 使用签名 tag、SHA256、依赖锁和 SBOM；
4. 文档要求用户固定 `revision`，不要从浮动 `main` 执行；
5. 长期向 Transformers 上游提交 `bidirectional/full attention token classification` 支持，或发布稳定 PyPI 包以降低远程代码依赖。

Hugging Face 文档明确指出自定义模型可能需要执行远程代码；同时 `safetensors` 相比 pickle 权重更安全且加载更快。因此公开包只提供 `safetensors`。[R32][R34]

### 11.6 Model Card 与 Dataset Card

模型卡至少包含：

- YAML metadata：`language: zh`、`pipeline_tag: token-classification`、`base_model`、`license`、`datasets`、`tags`；
- 模型架构和因果 mask 改造说明；
- 标签表、Presidio 映射、默认阈值和校准版本；
- 训练数据类别与比例，不披露客户内容；
- 判别式/生成式教师及其版本和许可；
- 每实体、每场景、噪声、长文、hard-negative 的结果与置信区间；
- 与 ERNIE、规则和 Presidio 端到端组合的结果；
- 已知限制：方言、OCR、嵌套实体、隐含身份、非文本、域外格式；
- 隐私风险、记忆测试、失败模式和“不是合规保证”的声明；
- 自定义代码安全说明和固定 revision 的加载示例；
- citation、联系方式、漏洞报告流程。

Dataset Card 必须记录生成器版本、模板族、实体值来源、去重策略、许可、字段、拆分方法、已知偏差和真实 PII 扫描结果。Hugging Face 建议模型卡明确用途、限制、训练数据和评测结果。[R33]

### 11.7 Presidio 开源包

GitHub 发布可安装模块：

```text
pip install pii-zh-qwen[presidio]
```

公共 API：

```python
from pii_zh.presidio import build_qwen_recognizer

recognizer = build_qwen_recognizer(
    model_id="<org>/zh-pii-qwen3-0.6b-bi",
    revision="v1.0.0",
    thresholds="builtin://balanced-v1",
)
```

提供三套示例策略：

- `high_recall`：安全审计/数据外发；
- `balanced`：通用客服和日志；
- `high_precision`：内容展示和搜索索引。

通用身份证、银行卡、电话、邮箱等规则可以公开；客户名称、业务字段、内部单号和 allowlist 只提供配置模板，不提交真实规则值。

### 11.8 开源发布门禁

每次 Hugging Face/GitHub release 必须全部通过：

#### 法务和来源

- 基础模型、每个数据源、每个教师来源均有许可证/条款记录；
- API 教师有允许公开训练与分发的书面授权，否则确认未进入训练；
- `NOTICE`、署名和第三方许可证完整；
- 未混入 NC、ND、仅研究或来源不明数据。

#### 隐私

- 对模型仓库、GitHub 历史、样本、测试、日志做真实 PII 扫描；
- 对训练数据做重复/近重复与客户模板泄漏检查；
- 运行 canary extraction、membership inference 风险检查和异常记忆探测；
- 人工复核模型卡与示例，确保没有真实个人信息。

#### 质量

- 冻结公共测试、私有跨客户测试、hard negative、噪声和长文测试全部达标；
- 至少 3 个种子；
- 结果可从 clean checkout 重现；
- Presidio 端到端回归通过；
- 公开阈值与报告中的阈值一致。

#### 安全与供应链

- 只发布 `safetensors`；
- 自定义远程代码经过静态扫描和人工 review；
- tag、commit、权重 checksum 和 SBOM 一致；
- 依赖无已知高危漏洞或已记录例外；
- SECURITY.md 提供私密漏洞报告渠道。

#### 可复现性

- 发布训练配置、随机种子、数据 manifest hash、tokenizer hash、base revision、teacher revision；
- 发布评测脚本和一个不含真实 PII 的小型 smoke-test 数据集；
- 私有金标不能公开时，明确区分公开结果与内部结果，并公开可复现的公共评测部分。

---

## 12. 部署方案

### 12.1 服务形态

推荐把 Qwen recognizer 作为 Presidio analyzer 进程内的单例组件：

- 启动时加载一次；
- 请求按长度做动态 batch；
- tokenizer-aware 切窗；
- model、calibrator、规则和策略分别热更新或滚动发布；
- 设置明确超时和 fallback；
- Qwen 超时时仍运行结构化规则和现有 ERNIE。

### 12.2 推理优化顺序

1. BF16 + SDPA 基线；
2. micro-batching；
3. `torch.compile`，先做结果 parity；
4. 更高效 attention kernel，确认 `is_causal=False`；
5. ONNX/TensorRT 或其他后端；
6. INT8/权重量化；
7. 8B/14B → 0.6B 蒸馏。

任何优化都必须重新跑严格 span、每类 recall、校准和端到端泄漏测试。

### 12.3 Shadow 和灰度

#### Shadow

- Qwen 结果不参与 anonymizer；
- 记录脱敏后的统计：命中类别、分数桶、与 ERNIE/规则的一致性；
- 原始文本和实体值不进入普通监控；
- 在安全环境抽样人工复核 disagreement。

#### Canary

- 先启用 Qwen 独有且高收益类别；
- 再启用 PERSON/ADDRESS 等重叠类别；
- 按客户和场景逐步放量；
- 每个策略版本可一键回滚到 ERNIE + rules。

#### 全量

- 达到门槛后，可移除 ERNIE 的部分重叠实体；
- 是否完全移除 ERNIE 由端到端增益和资源成本决定，不以模型新旧决定。

---

## 13. 监控与主动学习

### 13.1 在线监控

只记录聚合或脱敏后的指标：

- 每类命中率和分数分布；
- Qwen/ERNIE/规则 disagreement；
- checksum-valid 规则未被模型识别的比例；
- 模型独有命中比例；
- 文本长度和切窗数；
- p50/p95/p99 延迟；
- 超时、OOM 和 fallback；
- 客户/场景分层的误报申诉；
- 数据漂移：实体类别、格式、语言和噪声分布。

### 13.2 主动学习采样

优先采样：

- 高熵/低 margin；
- Qwen 与 ERNIE 冲突；
- 模型与 checksum 规则冲突；
- 模型独有高分 span；
- 规则独有但模型漏检；
- 客户误报/漏报反馈；
- 新格式、长文本、噪声和新场景；
- 允许列表/拒绝列表频繁触发的实体。

每轮主动学习数据必须进入安全标注、重新分组切分，并运行全量回归，而不是只在新增样本上验证。

---

## 14. 安全、隐私和合规控制

- 真实 PII 训练数据不出受控环境；
- 第三方 LLM API 不接收未经不可逆去标识化的真实 PII；
- 所有教师来源记录模型版本、部署方式、许可证/合同版本、prompt hash 和允许用途；
- “输出归用户所有”不等于“允许训练并公开分发派生模型”，必须单独审查竞争模型与分发条款；
- API 条款发生变化时冻结新调用，重新评审后再恢复；历史输出按调用时适用合同和订单条款追踪；
- 公开 release 前生成训练数据来源的可达性清单，确保可以证明每条样本来自允许的来源；
- 公开模型发布前确认权重、样本、日志和 model card 不含真实值；
- 不因模型是 Token Classification 就假设没有记忆风险；
- 使用 canary 和 membership/inversion 风险测试；
- 客户数据和模型训练授权单独记录；
- 跨客户数据训练必须满足合同和合规要求；
- 模型、LoRA、optimizer state、梯度和缓存均视为敏感产物；
- 所有数据集保留许可、来源、修改、过滤和归属说明；
- CC-BY 数据保留署名；NC 数据不得进入商业发布链路；
- 外部测试集和内部冻结测试集权限隔离；
- 规则和场景配置中不得明文存放真实 PII 允许列表，必要时使用 hash/安全配置中心。

---

## 15. 实施阶段与退出条件

### 阶段 A：规范和现网基线

交付：

- 标签体系 v1；
- 标注指南；
- 数据合同；
- 现网 ERNIE + rules 的模型级和端到端报告；
- 测试集隔离策略。

退出条件：所有 span 定义、冲突规则、风险层和 Presidio mapping 经过业务/安全评审。

### 阶段 B：数据流水线

交付：

- OpenPII 转换器；
- 自建合成生成器；
- offset/token 对齐；
- group split 和近重复检测；
- 数据质量/许可报告；
- `pii-bench-zh` 冻结评测入口。

退出条件：所有样本通过 span 校验；泄漏检查、模板隔离和许可检查通过。

### 阶段 C：Qwen 方法选型

交付：

- causal、Full Attention、JPT 三条基线；
- Full Attention 正确性测试；
- 0.6B LoRA/全量结果；
- tokenizer 边界审计；
- 初步性能报告。

退出条件：Full Attention 或 JPT 在内部验证上稳定优于 causal，且部署复杂度可控。

### 阶段 D：金标、教师阶梯和蒸馏

交付：

- 内部金标 v1；
- 4B 低成本教师消融；
- 8B 默认判别式教师；
- 14B 判别教师上界实验；
- 自托管 Qwen3.6-27B 与 35B-A3B 生成式标注/仲裁实验；
- 0.6B hard-span、token-KL 和多教师蒸馏模型；
- 1.7B 质量学生对照；
- teacher/data provenance、API 条款快照、calibration 和类别阈值；
- 跨客户/场景/时间评测与单位合格标注成本报告。

退出条件：教师自身达到准入门槛；蒸馏后的学生在至少 3 个种子上稳定提升；14B 是否保留由相对 8B 的学生增益决定；最差客户/场景没有不可接受退化。

### 阶段 E：Presidio 融合

交付：

- `QwenPiiRecognizer`；
- tokenizer-aware chunker；
- 中文上下文增强器；
- 通用/行业/客户规则包；
- 融合器；
- end-to-end benchmark。

退出条件：Qwen + rules 至少在一个关键维度明显优于当前 ERNIE + rules，且整体过度脱敏不恶化。

### 阶段 F：Shadow、Canary、全量

交付：

- 在线监控；
- disagreement 抽样；
- 主动学习流水线；
- 回滚方案；
- 生产 model card 和安全评审。

退出条件：端到端指标、SLA、稳定性、数据保护和回滚演练全部通过。

### 阶段 G：开源审计与发布

交付：

- Hugging Face 模型仓库；
- GitHub 训练/评测/Presidio 代码仓库；
- 可选的公开合成数据集仓库；
- LICENSE、NOTICE、THIRD_PARTY_NOTICES、SBOM 和 provenance；
- 公开可复现评测、签名 tag 和 checksum；
- `trust_remote_code` 安全审计及上游 Transformers 路线图。

退出条件：法务、隐私、安全、质量和可复现性门禁全部通过；公开仓库扫描不到真实 PII；API 教师来源均有明确公开训练与公开分发授权，或能通过 manifest/重训证明其从未进入公开权重；0.6B/1.7B 每个公开检查点均有独立 provenance。

---

## 16. 风险清单与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| decoder 因果上下文不足 | 人名/地址边界和歧义错误 | Full Attention 主线，JPT 对照 |
| Full Attention patch 随 Transformers 变化 | 升级后静默退化 | 锁版本、未来 token 单测、保存自定义架构版本 |
| 合成模板泄漏 | 测试 F1 虚高 | template family group split，外部冻结测试 |
| 公共数据中文不自然 | 领域泛化差 | 人工抽检、降权、金标收敛 |
| Qwen tokenizer 跨实体边界 | 严格 span 误差 | 边界不可对齐审计、辅助 boundary refiner |
| 类别过多且长尾 | 稀有类召回差 | v1 限制核心标签、规则兜底、主动学习 |
| 模型与规则分数不可比 | 融合阈值失真 | 分实体 calibration、逻辑回归 stacking |
| 默认中文 context 不充分 | 误报/漏报 | 自定义 phrase-aware context enhancer |
| 过度遮盖普通日期、城市、组织 | 业务可用性下降 | 模型/策略分层、困难负样本、场景配置 |
| 客户规则污染通用模型 | 跨客户退化 | 规则外置、tenant 配置，不直接写入通用权重 |
| 真实 PII 进入日志或实验平台 | 数据泄露 | 安全环境、脱敏日志、审计和 canary |
| NC/不明许可数据进入商业模型 | 许可风险 | 数据 manifest 和 CI license gate |
| 8B/14B 教师训练成本过高 | 研发预算失控 | 教师只离线训练；先 8B、后 14B；以蒸馏学生收益决定是否保留 |
| 规则格式随政策/证件变化 | 漏检 | 规则包独立版本、定期回归和更新 |
| 平面 BIO 无法表达嵌套 | 丢失细粒度实体 | 原始嵌套 span 保留，Presidio 规则二次识别 |
| 商业 API 输出用于公开模型训练 | 条款违约、权重无法分发或被迫重训 | 默认不进公开池；逐合同书面授权；独立 shard；优先自托管开放权重 Qwen 教师 |
| 大教师参数更大但 PII 边界不更准 | 高成本、负蒸馏 | 冻结集准入、3 seeds、paired bootstrap；以蒸馏学生收益而非教师规模选型 |
| 生成式教师 hallucinate span/value | 错误伪标签污染训练 | JSON Schema/约束解码、start/end/quote 校验、独立 critic、多教师分歧转人工 |
| `trust_remote_code` 降低采用率或引入供应链风险 | 用户不愿加载、潜在远程执行风险 | 最小远程代码、固定 revision、签名/checksum、上游 Transformers PR |
| 开源仓库泄露客户 PII/模板 | 严重隐私与合同风险 | 独立 release 数据池、历史扫描、canary、人工复核、客户数据默认不公开 |
| 数据许可与模型许可证不兼容 | 权重无法按 Apache-2.0 发布 | public/private 双池、source registry、CI license gate、第三方 NOTICE、法务审批 |

---

## 17. 最小可执行清单

开始实现时按以下顺序执行：

- [ ] 冻结标签体系、风险层和 Presidio entity mapping；
- [ ] 建立当前 ERNIE + rules 的端到端测试基线；
- [ ] 下载并冻结 `pii-bench-zh`，设置只读测试权限；
- [ ] 建立统一字符 span JSONL 和校验器；
- [ ] 转换 OpenPII zh-CN，并完成 2,000 条质量审计；
- [ ] 实现中文合成生成器、困难负样本和模板隔离；
- [ ] 实现 Qwen token 对齐和不可对齐边界报告；
- [ ] 跑 Qwen3-0.6B causal baseline；
- [ ] 实现并单测 Qwen3 Full Attention；
- [ ] 跑 Full Attention、JPT、LoRA、全量微调消融；
- [ ] 建立内部金标和跨客户测试；
- [ ] 训练 4B 教师消融、8B 默认教师和 14B 上界；同时训练 1.7B 质量学生；
- [ ] 完成 tokenizer hash/offset 兼容性审计；仅兼容时运行 8B/14B → 0.6B/1.7B token-KL，否则运行字符 span 蒸馏；
- [ ] 仅在 14B 证明有学生增益后，运行 Qwen3-30B-A3B-Base MoE 教师研究分支并记录专家路由统计；
- [ ] 用自托管 Qwen3.6-27B 与 35B-A3B 做生成式标注/仲裁 A/B，并保留 Qwen3-30B-A3B-2507 回退；
- [ ] 建立 public/private/evaluation/quarantine 四类数据池、`teacher_provenance.jsonl` 和 API 教师合规门禁；
- [ ] 完成 calibration 和类别阈值；
- [ ] 实现 `QwenPiiRecognizer` 和 token-aware chunker；
- [ ] 实现中文上下文增强器、规则包和融合器；
- [ ] 跑 Rules/ERNIE/Qwen 全组合端到端评测；
- [ ] 完成 shadow、canary、回滚和主动学习闭环；
- [ ] 确认代码、模型、数据和教师来源的许可证兼容性；
- [ ] 对全部公开文件、Git 历史和样例运行真实 PII 扫描；
- [ ] 完成 canary extraction、记忆风险和供应链安全测试；
- [ ] 发布 Hugging Face 0.6B 模型；按门槛决定是否发布 1.7B；同步发布 GitHub 代码、模型卡、数据卡、SBOM、许可清单和评测报告。

---

## 18. 参考资料

以下资料均在 2026-07-10 检索；实际实施前应重新确认版本和许可证。

- **[R1] Qwen3 Base 模型卡**  
  https://huggingface.co/Qwen/Qwen3-0.6B-Base  
  https://huggingface.co/Qwen/Qwen3-1.7B-Base

- **[R2] Hugging Face Qwen3 文档：Qwen3ForTokenClassification**  
  https://huggingface.co/docs/transformers/en/model_doc/qwen3

- **[R3] Yano & Takamura, 2026. Treating Decoder-Only LLMs as Encoders: A Simple and Effective Fine-tuning Approach for Named Entity Recognition**  
  https://aclanthology.org/2026.bionlp-1.25/

- **[R4] Kukić et al., 2026. Sequence Repetition Enhances Token Embeddings and Improves Sequence Labeling with Decoder-only Language Models**  
  https://aclanthology.org/2026.findings-eacl.339/

- **[R5] Ewais et al., 2026. Just Pass Twice: Efficient Token Classification with LLMs for Zero-Shot NER**  
  https://aclanthology.org/2026.acl-long.526/

- **[R6] LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders**  
  https://arxiv.org/abs/2404.05961  
  https://github.com/McGill-NLP/llm2vec

- **[R7] Presidio TransformersNlpEngine 文档**  
  https://github.com/data-privacy-stack/presidio/blob/main/docs/analyzer/nlp_engines/transformers.md

- **[R8] Presidio：Using Transformers as an external PII model**  
  https://data-privacy-stack.github.io/presidio/samples/python/transformers_recognizer/

- **[R9] Presidio HuggingFaceNerRecognizer 源码**  
  https://github.com/data-privacy-stack/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/ner/huggingface_ner_recognizer.py

- **[R10] Presidio 发布记录/CHANGELOG**  
  https://github.com/data-privacy-stack/presidio/releases  
  https://github.com/data-privacy-stack/presidio/blob/main/CHANGELOG.md

- **[R11] OpenPII 1.5M**  
  https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m

- **[R12] PII Bench ZH**  
  https://huggingface.co/datasets/wan9yu/pii-bench-zh

- **[R13] CLUENER2020**  
  https://github.com/CLUEbenchmark/CLUENER2020  
  https://arxiv.org/abs/2001.04351

- **[R14] MultiPriv**  
  https://huggingface.co/datasets/CyberChangAn/MultiPriv

- **[R15] 中华人民共和国个人信息保护法**  
  https://www.cac.gov.cn/2021-08/20/c_1631050028355286.htm

- **[R16] 个人信息保护政策法规问答（敏感个人信息说明）**  
  https://www.cac.gov.cn/2026-01/09/c_1769688003183197.htm

- **[R17] GB/T 35273-2020 信息安全技术 个人信息安全规范**  
  https://std.samr.gov.cn/gb/search/gbDetailed?id=A0280129495AEBB4E05397BE0A0AB6FE

- **[R18] GB/T 42460-2023 信息安全技术 个人信息去标识化效果评估指南**  
  https://std.samr.gov.cn/gb/search/gbDetailed?id=F789206610ADB223E05397BE0A0AE533

- **[R19] Presidio Analyzer 架构与自定义 recognizer**  
  https://data-privacy-stack.github.io/presidio/analyzer/

- **[R20] Presidio 每实体上下文词讨论及当前限制**  
  https://github.com/data-privacy-stack/presidio/issues/1711

- **[R21] Qwen3.5-0.8B-Base**  
  https://huggingface.co/Qwen/Qwen3.5-0.8B-Base


- **[R22] Qwen3-8B-Base**  
  https://huggingface.co/Qwen/Qwen3-8B-Base

- **[R23] Qwen3-14B-Base**  
  https://huggingface.co/Qwen/Qwen3-14B-Base

- **[R24] Qwen3-30B-A3B-Instruct-2507**  
  https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507

- **[R25] Qwen3-235B-A22B-Instruct-2507**  
  https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507

- **[R26] Qwen3.5-35B-A3B-Base（混合 DeltaNet/MoE 架构参考）**  
  https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base

- **[R27] Qwen3.6-35B-A3B（后训练生成式教师候选及混合架构参考）**  
  https://huggingface.co/Qwen/Qwen3.6-35B-A3B

- **[R28] OpenAI Services Agreement（2025-12-01 更新，2026-01-01 生效）**  
  https://openai.com/policies/services-agreement/

- **[R29] Anthropic Commercial Terms 与输出训练说明（2026-03-16）**  
  https://www.anthropic.com/legal/commercial-terms  
  https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model

- **[R30] Gemini API Additional Terms of Service**  
  https://ai.google.dev/gemini-api/terms

- **[R31] Alibaba Cloud International Product Terms：Model Studio**  
  https://www.alibabacloud.com/help/en/legal/latest/alibaba-cloud-international-website-product-terms-of-service-v-3-8-0

- **[R32] Hugging Face Transformers：加载自定义模型与 `trust_remote_code`**  
  https://huggingface.co/docs/transformers/models

- **[R33] Hugging Face Model Cards**  
  https://huggingface.co/docs/hub/model-cards

- **[R34] Hugging Face Safetensors / Hub security**  
  https://huggingface.co/docs/safetensors/index
  https://huggingface.co/docs/hub/security-pickle

- **[R35] UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition**  
  https://arxiv.org/abs/2308.03279
  https://github.com/universal-ner/universal-ner

- **[R36] Resource-Efficient Anonymization of Textual Data via Knowledge Distillation from Large Language Models**  
  https://aclanthology.org/2025.coling-industry.20/

- **[R37] Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation**  
  https://aclanthology.org/2025.acl-long.139/

- **[R38] Qwen3.6-27B 官方模型卡**  
  https://huggingface.co/Qwen/Qwen3.6-27B

- **[R39] Qwen3.6-35B-A3B 官方模型卡**  
  https://huggingface.co/Qwen/Qwen3.6-35B-A3B

- **[R40] Qwen3.6 官方 GitHub：架构、Serving 与 Apache-2.0**  
  https://github.com/QwenLM/Qwen3.6

- **[R41] FiNERweb: Datasets and Artifacts for Scalable Multilingual Named Entity Recognition, EACL 2026 Findings**  
  https://aclanthology.org/2026.findings-eacl.121/

- **[R42] ZeroNER: Fueling Zero-Shot Named Entity Recognition via Entity Type Descriptions, ACL 2025 Findings**  
  https://aclanthology.org/2025.findings-acl.805/

- **[R43] 阿里云百炼：合规资质与隐私说明**  
  https://help.aliyun.com/zh/model-studio/privacy-notice

- **[R44] 阿里云百炼服务特别说明（模型体验输入与生成内容限制）**  
  https://help.aliyun.com/zh/model-studio/bailian-service-notes

- **[R45] Qwen3 官方 Hugging Face Collection（Base 检查点清单）**  
  https://huggingface.co/collections/Qwen/qwen3

- **[R46] Qwen3-30B-A3B-Base**  
  https://huggingface.co/Qwen/Qwen3-30B-A3B-Base

- **[R47] Hugging Face Transformers：Qwen3MoE 与 `Qwen3MoeForTokenClassification`**  
  https://huggingface.co/docs/transformers/en/model_doc/qwen3_moe

---

## 19. 最终建议

面向最终开源，教师体系应定为“两种教师、两种监督、两条数据链”：

```text
公开高性价比学生：Qwen3-0.6B-Base Full-Attention Token Classification
公开质量学生（条件发布）：Qwen3-1.7B-Base Full-Attention Token Classification
默认判别教师：Qwen3-8B-Base Full-Attention Token Classification
Dense 判别教师上界：Qwen3-14B-Base Full-Attention Token Classification
MoE 研究上界：Qwen3-30B-A3B-Base（非默认，约 3.3B active/token）
默认生成式教师：自托管 Qwen3.6-27B（text-only，字符 span JSON）
高吞吐生成教师：自托管 Qwen3.6-35B-A3B
兼容性回退：Qwen3-30B-A3B-Instruct-2507
第三方商业 API：默认不进入 public_release_pool；书面授权后才可例外
```

### 对“是否使用更大教师”的明确回答

**应该使用更大教师做实验，默认从 8B 判别教师开始，14B 做 dense 上界；必要时再把 30B-A3B-Base 作为 MoE 研究上界，同时用 27B/35B 生成模型扩大标注覆盖。** 但不要把“更大”简化为一个模型：

- 8B/14B 判别教师与学生做同一 Token Classification 任务，可提供 logits、BIO 和 span；
- 30B-A3B-Base 也可做 Token Classification，但每 token 只激活约 3.3B 参数，必须单独评估专家路由和 Full-Attention 改造，不能直接推断一定优于 14B dense；
- Qwen3.6-27B/35B-A3B 生成教师负责候选标注、二次审校、难例解释和冲突仲裁，只提供字符级监督；
- 规则和 ERNIE 继续提供独立证据；
- 人工金标负责校正所有教师的系统性偏差。

决定 14B 是否值得的指标不是教师自身成绩，而是：

```text
14B 蒸馏学生的端到端增益
- 额外训练/推理成本
- hard-negative 误报风险
- 开源和复现复杂度
```

若 14B 相对 8B 不能使学生 Strict span F1 稳定提升约 0.2–0.3 个百分点，或没有改善 Tier-0 Recall、域外场景和最差组，就停止扩容。

### 对“是否使用 LLM API”的明确回答

**技术上可以，开源目标下默认不建议依赖。** API 通常无法提供与学生输入 token 对齐的完整 BIO logits，主要价值是高质量候选标注和仲裁；更重要的是，各供应商对训练、竞争模型、数据处理和第三方分发的限制不同。OpenAI 的公开协议只在开发“与 OpenAI 竞争的 AI 模型”时触发限制，但其分类器例外又排除了第三方分发或商业提供；公开条款并未替你判定 PII Token Classifier 是否属于竞争模型。因此这不是可以自行下结论的“当然允许”或“当然禁止”，而应在用于公开权重前取得针对具体项目的书面确认。Anthropic、Gemini 等服务同样应按调用时的具体合同和产品版本审核。[R28][R29][R30]

因此执行顺序应为：

```text
自托管开放权重 Qwen 教师
> 有许可证的公共/合成数据
> 多教师一致 + 本地校验器
> 人工金标与主动学习
> 通过书面开源授权的付费 API
> 授权不明的 API（仅内部评测，绝不训练公开权重）
```

### 最终发布建议

1. 先发布 `zh-pii-qwen3-0.6b-bi`，保证普通 GPU/CPU 量化场景可用；
2. 只有 1.7B 在 Presidio 端到端上带来明确收益时，再发布质量档；
3. 公共模型只从 `public_release_pool` 训练；客户金标、客户规则和私有 LoRA 留在 `private_enterprise_pool`；
4. 公开模型卡同时报告“模型单独”和“Qwen + Rules + ERNIE + Presidio”结果；
5. 发布 `teacher_provenance.json`、`data_provenance.json`、训练 manifest hash、tokenizer 兼容性报告和无 API 重训 recipe；
6. 自定义 Full-Attention 代码保持最小化、只发布 safetensors、要求固定 revision，并推进 Transformers 上游集成；
7. 不把开源模型描述为合规保证，而是明确其适用范围、已知漏检和推荐规则包。

生产系统长期保持：

```text
Qwen 语义模型
+ ERNIE 辅助
+ 中国大陆格式/校验位规则
+ 场景与客户策略
+ Presidio 编排、校准和审计
```

开源模型提供通用简体中文语义检测能力；客户差异继续通过外置规则、allowlist、上下文、阈值和私有 adapter 实现，而不是把客户专有信息写入公共权重。
