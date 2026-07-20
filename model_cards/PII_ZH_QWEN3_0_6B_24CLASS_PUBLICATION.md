---
language:
- zh
library_name: transformers
pipeline_tag: token-classification
license: apache-2.0
base_model: ZJUICSR/AIguard-pii-detection-fast
base_model_revision: 677a5ebc1600fef61e8973cafd3026be322b3a73
tags:
- pii
- chinese
- token-classification
model_name: pii-zh-qwen3-0.6b-24class
package_version: 0.2.0rc1
publication_state: release_candidate
---

# pii-zh-qwen3-0.6b-24class

面向简体中文的本地 PII token-classification 模型，覆盖 24 个实体类型和 49 个 BIO token
标签。模型权重采用 safetensors 格式，可直接通过 Transformers 使用，也可以作为中文规则、
Presidio 与模型级联服务中的语义识别组件。

## 30 秒开始

Python 3.10 及以上环境可直接安装推理依赖：

```bash
python -m pip install "torch>=2.4,<3" "transformers>=5.13,<6"
```

下面的示例从 Hugging Face 下载固定版本并在本机 CPU 上推理。输入为虚构数据；输出仅保留
标签、字符位置和分数，不打印命中的原文。

```python
from transformers import pipeline

model_id = "Forrest20231206/pii-zh-qwen3-0.6b-24class"
detector = pipeline(
    "token-classification",
    model=model_id,
    tokenizer=model_id,
    revision="v0.2.0rc1",
    trust_remote_code=True,
    aggregation_strategy="simple",
    device=-1,
)

text = "联系人张三，邮箱 demo@example.com，电话 13800138000。"
spans = detector(text)
safe_output = [
    {
        "label": span["entity_group"],
        "start": span["start"],
        "end": span["end"],
        "score": float(span["score"]),
    }
    for span in spans
]
print(safe_output)
```

该示例会识别 `PERSON_NAME`、`EMAIL_ADDRESS` 和 `PHONE_NUMBER`。首次运行需下载约 2.4 GB
权重。`trust_remote_code=True` 会加载仓库内公开的模型实现，正式部署建议固定版本并先审阅代码。

## 模型规格

| 项目 | 说明 |
|---|---|
| 参数量 | 596,100,145（约 0.6B） |
| 权重 | fp32 safetensors，约 2.38 GB |
| 架构 | Qwen3 全双向注意力 token classifier |
| 输出 | `O + 24×BIO`，共 49 个 token 标签 |
| 初始化模型 | `ZJUICSR/AIguard-pii-detection-fast` |
| 许可证 | Apache-2.0 |

初始化模型固定为 revision `677a5ebc1600fef61e8973cafd3026be322b3a73`。详细的第三方来源、
版本和许可证见 `THIRD_PARTY_NOTICES.md`。

## 24 类标签

| 类别 | 实体类型 |
|---|---|
| 基本信息与联系方式 | `PERSON_NAME`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, `ADDRESS`, `DATE_OF_BIRTH` |
| 身份证件 | `CN_RESIDENT_ID`, `PASSPORT_NUMBER`, `DRIVER_LICENSE_NUMBER`, `SOCIAL_SECURITY_NUMBER` |
| 金融账户 | `BANK_CARD_NUMBER`, `BANK_ACCOUNT_NUMBER`, `ALIPAY_ACCOUNT` |
| 车辆、组织与医疗 | `VEHICLE_LICENSE_PLATE`, `EMPLOYEE_ID`, `STUDENT_ID`, `MEDICAL_RECORD_NUMBER` |
| 社交账号、网络与设备 | `WECHAT_ID`, `QQ_NUMBER`, `USERNAME`, `IP_ADDRESS`, `MAC_ADDRESS`, `DEVICE_ID` |
| 位置与安全信息 | `GEO_COORDINATE`, `SECRET` |

`SECRET` 表示密码、密钥等安全敏感信息，不一定属于法律定义中的个人信息。模型包中的
`taxonomy.yaml`、`id2label.json` 和 `config.json` 提供完整且一致的标签定义。

## 评测结果

### Open-24 同协议合成评测

Open-24 是覆盖 24 类、包含无 PII 文档的确定性合成评测。下表中的单模型与完整级联服务属于
不同轨道：只加载本模型时，只能引用“单模型”结果。

| 轨道 | 系统 | strict-span micro F1 | strict-span micro recall | 无 PII 文档 FPR |
|---|---|---:|---:|---:|
| 单模型 | 本模型 | **0.9867** | **0.9907** | **0.0000** |
| 单模型对照 | AIguard fast 原始模型 | 0.2512 | 0.1862 | 0.1160 |
| 集成服务 | 本项目级联服务 | **0.9888** | 0.9781 | **0.0000** |
| 服务对照 | Presidio + CLUENER + cn_common_v6 | 0.6237 | 0.5185 | 0.1622 |

这些数字来自合成协议，不能代表真实业务、人类盲测或跨域文本的效果。

### PII Bench ZH 公开测试

PII Bench ZH 是 100% 程序化合成的公开测试集，不包含无 PII 文档。当前仅评测了单模型，
没有评测完整级联服务。

| 子集 | 文档数 | strict micro F1 | strict macro F1 |
|---|---:|---:|---:|
| Formal | 5,000 | 0.5992 | 0.5799 |
| Chat | 3,000 | 0.4763 | 0.4187 |
| Pooled | 8,000 | 0.5575 | 0.5297 |

在同一 closed-8 benchmark 的现有单模型结果中，OpenMed QwenMed XLarge 600M 的 pooled
micro F1 为 0.6239，OpenMed Privacy Filter Multilingual v2 为 0.6775，AIguard raw
envelope 为 0.7364。本模型在该公开测试上并非领先。由于生成方式、解码器和标签覆盖不同，
这些结果适合做参考，不应解释为无条件排行榜。

## 训练数据

- 训练集 87,995 条，其中无 PII 文档 39,600 条；
- 开发验证集 2,005 条，其中无 PII 文档 900 条；
- 两个集合均覆盖全部 24 类；
- 数据为 100% 确定性合成和开源派生数据，不含客户数据、生产数据或真实 PII；
- 模板资产包括 53 个经人工筛选的 Qwen3-8B 辅助模板、34 个人工正例模板和 20 个人工
  hard-negative 模板。

训练集和开发验证集共享 17 个模板家族，因此开发集与 Open-24 结果可能高估对新模板和真实
业务文本的泛化能力。

## 级联服务

结构化标识更适合规则和校验器，姓名、地址等上下文实体更依赖模型。完整项目提供中文规则、
Presidio 兼容层、确定性融合、CLI 和 FastAPI 服务。安装与部署方式见
[GitHub 项目](https://github.com/whyiug/pii-detect-model)。模型结果与级联服务结果不可混用。

## 适用范围与限制

- 适用于简体中文文本中的候选 PII span 检测，不直接提供法律或合规结论；
- 真实部署前应使用本业务域数据重新评估标签定义、阈值、误报率和漏报率；
- 长文档建议先分块，或使用项目提供的窗口化级联服务；
- 模型可能漏检、误检或输出输入中的敏感片段，应用日志不应保存原文和命中值；
- 自动检测不能替代访问控制、数据最小化和必要的人工复核。

安全问题请通过
[GitHub 安全报告页面](https://github.com/whyiug/pii-detect-model/security/advisories/new)提交，
不要在公开 issue 中附带真实 PII、凭据或可被直接滥用的细节。

## 项目与版本

- `github_repository: whyiug/pii-detect-model`
- `hugging_face_repository: Forrest20231206/pii-zh-qwen3-0.6b-24class`
- `git_source_commit: {{GIT_SOURCE_COMMIT}}`
- `release_tag: {{RELEASE_TAG}}`
- `github_release_url: {{GITHUB_RELEASE_URL}}`
