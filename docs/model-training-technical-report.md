# pii-zh-qwen3-0.6b-24class 数据集与训练技术报告

> 文档版本：1.0
>
> 对应模型：[`Forrest20231206/pii-zh-qwen3-0.6b-24class`](https://huggingface.co/Forrest20231206/pii-zh-qwen3-0.6b-24class)
>
> 对应发布：`v0.2.0rc1`
>
> 文档性质：简要工程技术报告，不是论文，也不构成真实业务效果或合规保证。

## 摘要

`pii-zh-qwen3-0.6b-24class` 是一个面向简体中文的 24 类 PII token-classification
模型。它以固定 revision 的 `ZJUICSR/AIguard-pii-detection-fast` 为初始化来源，将 Qwen3
骨干从因果注意力转换为 padding-aware 的双向全注意力，并把输出空间统一为
`O + 24×BIO = 49` 个标签。模型使用 87,995 条训练样本和 2,005 条开发验证样本进行三 seed、
两轮 LoRA 微调，最终选择 seed 97，并将 LoRA 权重合并为可独立加载的 FP32 safetensors 模型。

训练和开发数据均为确定性程序生成的合成数据，不含客户数据、生产数据或已知真实 PII。模板设计
包含本地 Qwen3-8B 辅助提出并经人工筛选的模板、人工编写的正例模板和 hard-negative 模板。
该数据具有可审计、可哈希和字符级标注精确的优点，但不能替代真实业务语料；尤其是训练集与开发
验证集存在 17 个模板组重叠，因此开发集结果不能作为独立泛化证据。

## 1. 模型概况

| 项目 | 内容 |
|---|---|
| 任务 | 简体中文 PII 字符 span 检测 / token classification |
| 架构 | Qwen3 双向全注意力 token classifier |
| 参数量 | 596,100,145（约 0.6B） |
| 标签空间 | 24 个实体类型，49 个 BIO 标签 |
| 最大训练长度 | 512 tokens |
| 初始化模型 | `ZJUICSR/AIguard-pii-detection-fast` |
| 初始化 revision | `677a5ebc1600fef61e8973cafd3026be322b3a73` |
| 最终训练方式 | LoRA 微调后合并为独立模型 |
| 最终 seed | 97 |
| 权重格式 | 单文件 FP32 safetensors，约 2.38 GB |
| 代码/模型许可证 | Apache-2.0；第三方条款见项目 `THIRD_PARTY_NOTICES.md` |

24 个实体类型及其稳定顺序由
[`taxonomy.yaml`](../src/pii_zh/taxonomy/taxonomy.yaml)定义，包括姓名、电话、邮箱、地址、
身份证件、金融账号、医疗/教育/员工标识、社交账号、网络设备标识、坐标和安全敏感字符串等类别。

## 2. 训练数据

### 2.1 模板如何产生

训练语料不是由大模型直接生成标签，也不是从真实 PII 文本中抽取。数据构建分为三个阶段：

1. **模板设计与筛选**：本地 `Qwen/Qwen3-8B` 以确定性解码方式提出只包含占位符的候选模板。
   70 个候选经人工语义、标签和近重复审查后接受 53 个、拒绝 17 个。Qwen3-8B 没有接触最终
   实体值，也没有为训练样本做 span 标注或伪标签。
2. **人工补充**：另外编写 34 个正例模板和 20 个 hard-negative 模板，用于补充场景覆盖，
   以及区分配置键、任务编号、产品型号等“表面像标识符但不是 PII”的文本。
3. **确定性实例化**：版本化生成器为占位符构造合成实体值，渲染文本并计算精确字符 offset；
   身份证件、银行卡、车牌等结构化类别还需通过对应格式校验器。字符 span 再通过 fast tokenizer
   的 offset mapping 转为 BIO 标签，padding 和特殊 token 使用 `-100` 忽略。构建时固定代码、
   配置标识与 seed，并记录输出 SHA-256。

完整模板资产共有 87 个正例模板和 20 个 hard-negative 模板；最终训练/开发池实际使用其中
91 个模板组：45 个 Qwen3-8B 辅助提出并经人工接受的正例、30 个人工正例和 16 个人工
hard-negative。其余 16 个历史 test 角色模板没有进入权重训练。
模板资产和最终数据集配置均声明 Apache-2.0。模板、人工筛选结论和生成器在仓库中版本化：

- [`curated_templates_v1.json`](../src/pii_zh/data/synthetic/assets/curated_templates_v1.json)
- [`curation_audit_v1.json`](../src/pii_zh/data/synthetic/assets/curation_audit_v1.json)
- [`sota_v1.py`](../src/pii_zh/data/synthetic/sota_v1.py)

合成实体值由本地工厂生成，不从真实个人数据采样。邮箱、IP、MAC 等部分字段使用文档或测试
命名空间；手机号、日期等字段仍保持真实格式，因此这里的准确表述是“无真实数据来源”，而不是
“任何生成值都绝不可能与现实标识偶合”。

### 2.2 最终训练/开发集规模

最终数据集标识为 `pii_zh_synthetic_sota_v2_resplit` 2.0.0，总计 90,000 条文本。
这些文本在原始 90,000 条语料内全文唯一，合成实体值组也全局唯一。

| 统计项 | 训练集 | 开发验证集 |
|---|---:|---:|
| 文档数 | 87,995 | 2,005 |
| 含 PII 文档 | 48,395 | 1,105 |
| PII-free 文档 | 39,600 | 900 |
| PII-free 占比 | 45.0% | 44.9% |
| 实体数 | 104,463 | 2,890 |
| tokenizer token 数 | 3,813,934 | 96,754 |
| 覆盖实体类型 | 24 | 24 |
| 无法对齐的字符边界 | 0 | 0 |

训练集中不同标签的实体频次并不均衡，例如 `PERSON_NAME` 明显多于部分结构化类别。最终训练配方
没有使用 class weighting 或 document re-sampling，而是保留原始确定性分布。

数据文件通过 SHA-256 固定：

| 文件 | SHA-256 |
|---|---|
| `train.jsonl` | `19ad4684923dc9d05e72efb50e1444a5f02c538c0107e388503f685ab5dab01b` |
| `validation.jsonl` | `cc0c248aeba4ae53048a9bd0bda4c19c751a5eb4eff59bb4a885e5b05bf0949f` |
| dataset manifest 文件 | `3f0c348a29efa6e7c83d17261e589f1762151cca3898d78f3b5ed2a1553a8748` |
| dataset manifest logical | `a57f0719a9d225ee72a614e73e0b84f61be243e9fb4dfe6a56074015a8285186` |

### 2.3 划分方法和隔离边界

原始合成语料由 80,000 条 train 和 10,000 条 validation 组成。最终版本按照模板与 PII 状态进行
确定性哈希重划分，将旧 validation 中 7,995 条转入训练集，得到 87,995 / 2,005 的开发划分。
配置和实现见：

- [`synthetic_sota_v2_resplit.yaml`](../configs/data/synthetic_sota_v2_resplit.yaml)
- [`sota_v2_resplit.py`](../src/pii_zh/data/synthetic/sota_v2_resplit.py)

跨 split 检查结果为：

- `document_id` 碰撞：0；
- `entity_value_group` 碰撞：0；
- `source_group` 碰撞：0；
- `template_group` 重叠：17。

这意味着开发验证集没有复用相同文档或相同实体值，但部分模板语义家族在训练中已经出现。
该 validation 的角色是 checkpoint/seed 选择和开发诊断，不是独立盲测集。当前 GitHub 仓库公开
了模板与生成/重划分代码，Hugging Face training manifest 公开了聚合统计与文件哈希，但两者都
没有分发记录级训练 JSONL；因此外部用户目前不能只依赖一次 Git checkout 完成最终 90,000 条
语料的逐字节重建。

### 2.4 未用于训练的数据

公开 PII Bench ZH、CLUENER、`chinese_common_ner`、历史服务错误样本和生产文本均未用于该模型的
权重训练。PII Bench ZH 也未用于 validation、checkpoint/seed 选择或阈值拟合；其结果属于模型
冻结后的 post-hoc 描述性评测。

## 3. 模型初始化

初始化过程由 [`aiguard24.py`](../src/pii_zh/models/aiguard24.py)实现，主要包含：

1. 固定并校验 AIguard source revision 与核心文件哈希；只接受 safetensors 权重。
2. 严格复制 596,049,920 参数的 Qwen3 backbone。
3. 将源模型中可兼容的 12 个实体分类头投影到目标 24 类 BIO 空间，同时复制 `O` 标签行；
   其余 12 类保留由固定 seed 产生的初始化。
4. 将 causal attention 转换为 padding-aware full attention。转换过程严格加载同一 state dict，
   没有丢弃参数或新增未初始化的 backbone 参数。

双向注意力实现在 [`qwen3_bi.py`](../src/pii_zh/models/qwen3_bi.py)。它使用 4-D additive mask，
禁用 KV cache，并支持 eager/SDPA attention backend。双向上下文更适合 token classification，
但不应把该模型当作自回归生成模型使用。

## 4. 训练配方

最终配方由
[`aiguard24_sota_v3_resplit_selection.json`](../configs/train/aiguard24_sota_v3_resplit_selection.json)
固定，训练入口为 [`train_aiguard24_sota.py`](../scripts/train_aiguard24_sota.py)。

| 超参数 | 设置 |
|---|---|
| candidate seeds | 13、42、97 |
| epochs | 2 |
| max length | 512 |
| train / eval batch size | 64 / 128 |
| gradient accumulation | 1 |
| backbone learning rate | `2e-5` |
| classifier learning rate | `2e-4` |
| optimizer | `adamw_torch` |
| LR scheduler | cosine，warmup ratio `0.05` |
| weight decay / max grad norm | `0.01` / `1.0` |
| LoRA rank / alpha / dropout | 32 / 64 / 0.05 |
| LoRA target modules | `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj` |
| precision | BF16 |
| gradient checkpointing | 开启，`use_reentrant=False` |
| loss | token-level cross entropy；padding/特殊 token 使用 `-100` 忽略 |
| class weighting | 关闭 |
| document sampling | 关闭 |
| checkpoint metric | validation `risk_weighted_score` |

每个 seed 独立训练，并在每个 epoch 后运行字符级严格 span 评测。训练器按
`risk_weighted_score` 选择 seed 内最佳 checkpoint；跨 seed 再依次比较 risk-weighted score、
strict micro F1、strict macro F1 和 PII-free precision，最后才用 seed 数值打破平局。

最终选择 seed 97 的 `checkpoint-2750`，即第 2 个 epoch 结束时的 checkpoint。LoRA adapter 随后
合并到基础模型中，输出单一 `model.safetensors`，不要求推理端另外加载 PEFT adapter。

### 4.1 训练环境

公开 training manifest 记录的选中运行环境如下：

| 项目 | 记录值 |
|---|---|
| GPU | 单张 NVIDIA A800-SXM4-80GB |
| CUDA runtime | 12.8 |
| Python / PyTorch | 3.12.12 / 2.10.0+cu128 |
| Transformers / PEFT / Accelerate | 4.57.6 / 0.18.1 / 1.13.0 |
| manifest 时间跨度 | 约 23 分 30 秒 |
| PyTorch 记录的峰值显存分配 | 约 4.09 GiB |

这些资源数字只描述该次 LoRA 训练，不等同于其他环境中的稳定吞吐或显存承诺。

## 5. 选模、评测与结果归因

选中 seed 97 的开发验证 strict micro/macro F1 均为 1.0，但该数字来自 validation-informed 的
合成开发划分，并存在 17 个 template group overlap，不能解释为真实世界泛化能力。

两个更有解释价值、但仍有限制的结果是：

| 评测 | 轨道 | strict micro F1 | 解释边界 |
|---|---|---:|---|
| Open-24 internal evaluation | 单模型 `model_raw` | 0.9867 | 100% 合成、覆盖 24 类 |
| PII Bench ZH pooled closed-8 | 单模型 `model_raw` | 0.5575 | 公开合成测试，post-hoc，且不是该榜单最佳模型 |

完整服务在 Open-24 上的 0.9888 属于“模型 + `cn_common_v6` 中文规则 + 项目 fusion”的级联结果，
不是单模型训练结果，也不包含 Presidio 本身的执行贡献。更完整的比较器、指标和限制见
[`benchmark_landscape.md`](benchmark_landscape.md)。`v0.2.0rc1` 没有分发该冻结评测使用的
calibration bundle，因此 0.9888 是项目冻结回放结果，不是默认 CLI 可逐字节复现的指标。

模型选择后进行的阈值校准和级联策略验证不会反向修改该模型的权重。公开 PII Bench 结果也没有
用于重新选择 seed、checkpoint 或训练超参数。

## 6. 可复现身份

公开模型包包含自哈希的
[`training_manifest.json`](https://huggingface.co/Forrest20231206/pii-zh-qwen3-0.6b-24class/blob/v0.2.0rc1/training_manifest.json)，
记录 source revision、数据哈希、训练配方、运行环境、checkpoint 和输出文件绑定。

| 制品 | 标识 |
|---|---|
| training manifest logical SHA-256 | `fa19c6c352e9bbfc8c3ee7c0bec41077df1a44132da297f276d9caf4e4737751` |
| `training_manifest.json` 文件 SHA-256 | `db9e16036fbcc9668e6f15be44fa3d02a060589c09d6609ee919d916940c6ed4` |
| manifest 记录的运行时 Git revision | `15986a8be18c3c5baa7355cc84b41baf9a5bd3a6` |
| final `model.safetensors` SHA-256 | `407515ad2319c0c1e2167ede01b10b1e5d4430fc1caba44f9827300c563845ee` |
| label schema SHA-256 | `beb79fe5610cbc56829f93a93f00d853b5549b9f9dce50ea1e8e9af9917ab099` |
| training recipe SHA-256 | `2f9ed3044a40f9a99a9c4c8f68422f063e366bd677f072c348703f1344aa119e` |

建议复现实验时同时固定：source model revision、训练代码 revision、train/validation SHA-256、
taxonomy、tokenizer、seed 和完整 recipe，而不是只记录目录名或模型名称。

需要注意，manifest 记录的 Git revision 只绑定该 commit 已追踪的文件；该 tree 尚未包含本版本后来
公开的训练入口、AIguard24 初始化实现和 v3 冻结选择协议，因而不能逐字节绑定这些实现与协议。
manifest 仍绑定已解析的 recipe、数据/源模型哈希及初始化审计。再考虑记录级训练数据未公开，以及
GPU/算子未承诺 bitwise deterministic，现有材料支持方法复跑，但不足以承诺逐字节复现相同权重。

## 7. 已知限制与后续研究

- 训练和开发数据全部为合成数据；没有真实或人工逐条标注的中文 PII 语料参与训练。
- 模板虽经人工筛选，但语体、行业、方言、OCR/ASR 噪声和长文档覆盖仍有限。
- 训练/开发存在模板组重叠，开发指标可能显著高估新模板和真实域性能。
- 当前记录级训练数据未作为独立 Dataset 仓库发布，外部逐字节复现仍不完整。
- 当前只比较了 3 个 seed，并在一种 GPU 环境完成；尚缺少系统性的规模、数据量和架构消融。
- 训练长度为 512 tokens，不能直接证明更长文档的召回、延迟和显存表现。

如果后续撰写论文，优先应补充：许可清晰的人工标注盲测集、跨领域/时间/模板 holdout、记录级数据
或完整生成配方发布、模型与级联服务的消融实验、置信区间，以及真实部署下的误报率、漏报率和资源
测量。在这些证据补齐前，本文只支持“可审计的中文 PII 合成训练模型”这一有限结论。
