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
publication_state: staged_not_uploaded
---

# pii-zh-qwen3-0.6b-24class

这是一个面向简体中文个人信息识别的 token-classification 模型，覆盖 24 个实体类型；
分类头包含 `O + 24×BIO`，因此共有 49 个 token 标签。模型可与中文规则、Presidio 兼容层
组成级联服务。当前卡片处于 `staged_not_uploaded` 状态，尚未上传到 Hugging Face。

## 发布目标与身份

- Python 包版本：`pii-zh-qwen==0.2.0rc1`
- GitHub 目标：`whyiug/pii-detect-model`
- Hugging Face 目标：发布前把下方唯一占位符渲染为登录后确认的 namespace
- `github_repository: whyiug/pii-detect-model`
- `hugging_face_repository: HF_NAMESPACE/pii-zh-qwen3-0.6b-24class`

以上仅是发布目标标识，不表示远端仓库、tag 或不可变 revision 已经存在。最终上传必须由外部、
经过校验的 `publication_manifest.json` 及其绑定回执授权；模型卡本身不承担发布授权，也不预写
Hugging Face revision。

## 使用方式

模型包采用 Transformers 兼容的 safetensors 格式。推荐通过 `pii-zh-qwen` 的字符级 span API
进行本地推理，并从经过 checksum 校验的模型目录读取权重。下面使用不含个人信息的合成文本；
输出只保留标签、字符位置与分数，不打印命中值。

```python
from pathlib import Path

from pii_zh.inference import load_local_predictor

model_dir = Path("./model-package").resolve(strict=True)
predictor = load_local_predictor(model_dir, device="cpu", micro_batch_size=1)
synthetic_text = "这是用于接口演示的合成文本。"
spans = predictor.predict(synthetic_text)
detections = [
    {
        "label": span["entity_type"],
        "start": span["start"],
        "end": span["end"],
        "score": span["score"],
    }
    for span in spans
]
print(detections)
```

如需验证原生 Transformers remote-code 兼容性，可以运行下面的本地最小前向。这一低层接口
只返回 token logits，不负责字符 span 解码。只应对来源和 checksum 均已审核的模型包启用
remote-code trust。

```python
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

model_dir = Path("./model-package").resolve(strict=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_dir, local_files_only=True, trust_remote_code=True
)
config = AutoConfig.from_pretrained(
    model_dir, local_files_only=True, trust_remote_code=True
)
model = AutoModelForTokenClassification.from_pretrained(
    model_dir, local_files_only=True, trust_remote_code=True
)
assert config.pii_attention_mode == "full"
assert config.num_labels == 49
model.eval()
inputs = tokenizer("这是合成测试文本。", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
assert logits.shape[-1] == 49 and torch.isfinite(logits).all().item()
print(tuple(logits.shape))
```

`config.json` 中的 `pii_release_eligible=false` 是冻结候选 lineage 与 remote-code 兼容合同的一部分，
不是当前发布授权，也不应由使用方改写。最终发布授权来自经过校验的外部 publication manifest、
许可证批准回执、私密安全渠道测试回执和本地候选回执。当前卡片及其模型包尚未上传。

## 冻结合成 Open-24 评测

以下数字仅对应本项目冻结、具名、100% 合成的 Open-24 同协议评测。质量聚合使用 10,000 次
按文档配对 bootstrap；结果不能外推到真实业务、人类盲测、跨域语料或生产部署。

| 轨道 | 系统 | strict-span micro F1 | strict-span micro recall | PII-free document FPR |
|---|---|---:|---:|---:|
| 单模型 | 本模型 | 0.9867 | 0.9907 | 0.0000 |
| 单模型对照 | `aiguard_pii_detection_fast_open24_model_raw_v1` | 0.2512 | 0.1862 | 0.1160 |
| 集成服务 | community model cascade | 0.9888 | 0.9781 | 0.0000 |
| 服务对照 | `native_presidio_cluener_cn_common_v6_open24_v1` | 0.6237 | 0.5185 | 0.1622 |

下表差值均为“本候选减具名比较器”；区间是按文档配对 bootstrap 的 95% 置信区间。

| 轨道 | 比较器 | Δ micro F1 | 95% CI | Δ macro F1 | 95% CI | Δ document FPR | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|
| 单模型 | `aiguard_pii_detection_fast_open24_model_raw_v1` | 0.7354 | [0.7260, 0.7446] | 0.8583 | [0.8537, 0.8631] | -0.1160 | [-0.1253, -0.1069] |
| 集成服务 | `native_presidio_cluener_cn_common_v6_open24_v1` | 0.3651 | [0.3570, 0.3733] | 0.5294 | [0.5253, 0.5336] | -0.1622 | [-0.1731, -0.1513] |

单模型与完整级联是不同轨道。仅加载权重得到的是 `model_raw`，不能引用完整级联服务指标。

## PII Bench ZH 与 OpenMed：已见公开测试

PII Bench ZH 是项目已经读取的、100% 程序化合成的公开测试集，且没有 PII-free 文档。下面是
冻结后的 `model_raw` 单次 post-hoc 描述性结果；`public_test_exposed=true`、
`selection_allowed=false`，没有运行当前完整级联服务。

| suite | documents | strict micro F1 | strict macro F1 | 完整级联服务 |
|---|---:|---:|---:|---|
| Formal | 5,000 | 0.59921050 | 0.57994984 | N/A（未评测） |
| Chat | 3,000 | 0.47628738 | 0.41874684 | N/A（未评测） |
| Pooled | 8,000 | 0.55750532 | 0.52965259 | N/A（未评测） |

在同一 closed-8 benchmark 的历史 `model_raw` 结果中，OpenMed QwenMed XLarge 600M 的 pooled
micro F1 为 `0.6239`，OpenMed Privacy Filter Multilingual v2 为 `0.6775`，已见 AIguard raw
envelope 为 `0.7364`。因此，本模型在该公开测试的单模型结果没有领先这些比较对象。不同
generation、decoder、标签覆盖和 framework hybrid 不能合并成无条件排行榜；这些已见结果也不得
用于继续选择模型、阈值或路由。

## 24 类标签

精确且有序的 24 类为：`PERSON_NAME`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, `ADDRESS`,
`DATE_OF_BIRTH`, `CN_RESIDENT_ID`, `PASSPORT_NUMBER`, `DRIVER_LICENSE_NUMBER`,
`SOCIAL_SECURITY_NUMBER`, `BANK_CARD_NUMBER`, `BANK_ACCOUNT_NUMBER`,
`VEHICLE_LICENSE_PLATE`, `EMPLOYEE_ID`, `STUDENT_ID`, `MEDICAL_RECORD_NUMBER`,
`WECHAT_ID`, `QQ_NUMBER`, `ALIPAY_ACCOUNT`, `USERNAME`, `IP_ADDRESS`, `MAC_ADDRESS`,
`DEVICE_ID`, `GEO_COORDINATE`, `SECRET`。

同一顺序由模型包内 `taxonomy.yaml`、`id2label.json` 和 `config.json` 共同约束。`SECRET` 是
安全敏感信息类型，不一定属于法律意义上的个人信息。

## 基座模型与标签迁移

固定初始化来源是 `ZJUICSR/AIguard-pii-detection-fast`，revision
`677a5ebc1600fef61e8973cafd3026be322b3a73`；其上游模型卡声明许可证为 Apache-2.0。
训练 manifest 记录了 backbone state-dict 复制、`O` 标签行复制，以及 12 个源实体分类头到
中文 core-24 标签的验证投影。源文件 SHA-256 与标签映射保存在模型包的
`training_manifest.json` 和 `THIRD_PARTY_NOTICES.md` 中。

## 训练数据事实

训练集共 87,995 条，其中 PII-free 39,600 条；开发验证集共 2,005 条，其中 PII-free
900 条。两个 split 均覆盖 24 类，并且均为 100% 确定性合成、开源派生数据，不含真实、
客户或生产 PII。模板资产包括本地 Qwen3-8B 辅助提出并经人工筛选接受的 53 个占位符模板、
34 个人工正例模板和 20 个人工 hard-negative 模板。

这不是盲验证：冻结 manifest 记录了 `validation_informed_template_family_overlap`，训练集与
开发验证集存在 17 个 template-group overlap。因此开发集与 Open-24 数字不能作为独立真实数据
泛化证据。

## 适用范围与限制

- 当前证据只覆盖具名合成评测与已见公开合成测试，不覆盖真实业务分布。
- 单模型不是唯一应用形态；正式集成建议评估 Presidio、中文规则与模型级联。
- 不同部署域需要重新评估标签定义、阈值、误报、漏报、文本长度与资源开销。
- 输出不得直接替代法律判断、合规控制、访问授权或人工复核。
- 安全问题应通过目标 GitHub 仓库的私密漏洞报告渠道提交；发布系统必须先用独立回执验证该渠道。

## 可复现性与发布合同

模型身份由模型 manifest、服务配置 manifest、校准 bundle、运行时源码清单、质量回执及两条
评测轨道的重放回执共同约束。publication successor 还必须绑定完整 Git source commit、目标
GitHub/Hugging Face repository ID、外部批准回执、重新生成的 checksum 闭包和
`publication_manifest.json`。

只有这些外部工件全部校验通过，远端上传与不可变 revision 才可执行。当前
`publication_state` 为 `staged_not_uploaded`，没有远端 revision 可供声明。
