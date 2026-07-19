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
publication_state: unpublished_local_candidate
---

# pii-zh-qwen3-0.6b-24class

这是一个面向简体中文个人信息识别的 token-classification 模型候选，覆盖 24 个实体类型；
分类头包含 `O + 24×BIO`，因此共有 49 个 token 标签。模型配套规则、Presidio 兼容层与
级联服务。当前材料仅用于本地发布前审核，尚未发布，也不作行业领先性、现实世界泛化或
生产可用性声明。

## 使用方式

安装版本固定为 `pii-zh-qwen==0.2.0rc1`。模型包采用 Transformers 兼容的
safetensors 格式；服务模式使用同一模型、冻结校准配置和社区级联 profile。加载模型时应
启用离线选项，并从已审核的本地模型目录读取权重。

推荐通过发行 wheel 的字符级 span API 做检测。下面的输入使用仅用于文档示例的保留域名
`example.com`，不代表真实联系人或项目收件地址；输出只保留标签、字符起止位置和分数，不打印
原文或命中值。实际返回以当前已校验模型包为准，示例不构成对任意输入必然命中的承诺。命令应从
`model-package` 所在目录运行。

```python
from pathlib import Path

from pii_zh.inference import load_local_predictor

model_dir = Path("./model-package").resolve(strict=True)
predictor = load_local_predictor(model_dir, device="cpu", micro_batch_size=1)
synthetic_text = "测试邮箱 demo@example.com"
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

如需验证原生 Transformers remote-code 兼容性，可运行下面的离线最小前向；这段低层示例返回
token logits，不负责字符 span 解码。只应对已经审核并校验过 checksum 的本地模型包启用
Transformers remote-code trust。

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
assert config.pii_release_eligible is False
assert config.pii_attention_mode == "full"
assert config.num_labels == 49
model.eval()
inputs = tokenizer("这是合成测试文本。", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
assert logits.shape[-1] == 49 and torch.isfinite(logits).all().item()
print(tuple(logits.shape))
```

## 已绑定评测

| 轨道 | 系统 | 状态 | strict-span micro F1 | strict-span micro recall | PII-free document FPR |
|---|---|---:|---:|---:|---:|
| 单模型 | 候选 | PASS | 0.9867 | 0.9907 | 0.0000 |
| 单模型 | 比较器 `aiguard_pii_detection_fast_open24_model_raw_v1` | 对照 | 0.2512 | 0.1862 | 0.1160 |
| 集成服务 | 候选 | PASS | 0.9888 | 0.9781 | 0.0000 |
| 集成服务 | 比较器 `native_presidio_cluener_cn_common_v6_open24_v1` | 对照 | 0.6237 | 0.5185 | 0.1622 |

下表差值均为“候选减比较器”；区间是按文档配对 bootstrap 的 95% 置信区间。

| 轨道 | 比较器 | Δ micro F1 | 95% CI | Δ macro F1 | 95% CI | Δ document FPR | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|
| 单模型 | `aiguard_pii_detection_fast_open24_model_raw_v1` | 0.7354 | [0.7260, 0.7446] | 0.8583 | [0.8537, 0.8631] | -0.1160 | [-0.1253, -0.1069] |
| 集成服务 | `native_presidio_cluener_cn_common_v6_open24_v1` | 0.3651 | [0.3570, 0.3733] | 0.5294 | [0.5253, 0.5336] | -0.1622 | [-0.1731, -0.1513] |

这些数字来自合成 Open-24 发布评测，并受独立重放门约束；它们不能替代真实业务数据、
跨域数据或人工标注盲测。Presidio、中文规则和大模型各有不同误报/漏报特征，推荐在应用侧保留
置信度、来源与审计信息。

### 附加公开测试已见结果（不参与选择或领先性声明）

PII Bench ZH 已被项目读取，下面是冻结后的 `model_raw` 单次 posthoc 描述性结果；公开测试暴露
为 `true`、`selection_allowed=false`，且没有通过描述性 active-comparator envelope。它与上面的
Open-24 轨道不是同一数据或 claim，不应混合比较。该次没有运行完整级联服务，因此服务结果为 N/A。

| suite | documents | strict micro F1 | strict macro F1 | full service |
|---|---:|---:|---:|---|
| Formal | 5000 | 0.59921050 | 0.57994984 | N/A（未评测） |
| Chat | 3000 | 0.47628738 | 0.41874684 | N/A（未评测） |
| Pooled | 8000 | 0.55750532 | 0.52965259 | N/A（未评测） |

## 24 类标签

精确且有序的 24 类为：`PERSON_NAME`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, `ADDRESS`, `DATE_OF_BIRTH`, `CN_RESIDENT_ID`, `PASSPORT_NUMBER`, `DRIVER_LICENSE_NUMBER`, `SOCIAL_SECURITY_NUMBER`, `BANK_CARD_NUMBER`, `BANK_ACCOUNT_NUMBER`, `VEHICLE_LICENSE_PLATE`, `EMPLOYEE_ID`, `STUDENT_ID`, `MEDICAL_RECORD_NUMBER`, `WECHAT_ID`, `QQ_NUMBER`, `ALIPAY_ACCOUNT`, `USERNAME`, `IP_ADDRESS`, `MAC_ADDRESS`, `DEVICE_ID`, `GEO_COORDINATE`, `SECRET`。同一顺序由模型包内 `taxonomy.yaml`、
`id2label.json` 和 `config.json` 共同约束；模型卡不会用 taxonomy 之外的泛化类别替代它们。

## 基座模型与标签迁移

本候选的固定初始化来源是 `ZJUICSR/AIguard-pii-detection-fast`，revision
`677a5ebc1600fef61e8973cafd3026be322b3a73`；其上游模型卡声明许可证为
`apache-2.0`。训练 manifest 记录了严格的 backbone state-dict
复制、`O` 标签行复制，以及 12 个源实体分类头到中文
core-24 标签的验证投影。具体源文件 SHA-256 与 12 类映射保留在模型包内
`training_manifest.json` 和 `THIRD_PARTY_NOTICES.md`。这份机械归属说明不替代发布前的人工
许可证复核。

## 训练数据事实与限制

训练集共 87,995 条，其中 PII-free
39,600 条；开发验证集共
2,005 条，其中 PII-free
900 条，两个 split 均覆盖 24 类。二者均为
100% 确定性合成、开源派生数据，不含真实、客户或生产 PII。模板资产包括本地 Qwen3-8B 辅助提出并
经人工筛选接受的 53 个占位符模板、
34 个人工正例模板和
20 个人工 hard-negative 模板。

这不是盲验证：冻结 manifest 明确记录 `validation_informed_template_family_overlap`，train/validation
存在 17 个 template-group overlap；因此开发集数字
不能作为独立真实数据泛化证据。

## 限制与安全边界

- 当前证据仅覆盖公开开源与合成数据，且公开测试集存在暴露风险。
- 单独模型效果不是唯一目标；正式应用建议使用 Presidio、规则与模型级联。
- 输出需要按场景复核，不应直接作为法律、合规或访问控制结论。
- 发布、tag、Hub 上传、签名与许可证批准均需另行授权和留痕。

## 可复现性

候选身份由最终模型 manifest、服务配置 manifest、完整运行时源码清单、校准 bundle、质量回执、
两条轨道的候选/比较器/性能重放回执共同绑定。模型卡本身处于
`unpublished_local_candidate`，不会预写 GitHub 或 Hugging Face 已发布状态。
