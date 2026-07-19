# 中文 PII 检测公开比较与发布定位

状态：2026-07-16 公开资料与仓库复现快照
用途：决定公开 benchmark、模型卡表述和隐藏榜协议；不是第三方结果的独立复现证明

## 结论先行

当前没有一个同时满足“简体中文、真实或人类自然语境、公开 gold、统一标签、统一严格字符 span
协议、隐藏测试、持续维护”的权威 PII 排行榜。可以公开比较，但必须分轨，不能把不同标签体系、
不同解码器、合成/真实语境和 model-only/full-system 分数放进同一排名。

本项目当前专用 train/validation/test 全部为确定性合成数据；外部 `PII Bench ZH` 也明确声明
100% 合成。现有发布候选的 headline 分数来自“模型 + 规则 + 融合 + validation 校准 + 后处理”
完整系统，不是可下载权重单独取得的分数。

“首个中文 PII 大模型”已经不是可维护的事实主张。公开可见的
[AIguard-pii-detection-fast](https://huggingface.co/ZJUICSR/AIguard-pii-detection-fast)
同样以 Qwen3-0.6B 为基座并覆盖 21 类中文实体；
[privacy-filter-tw](https://huggingface.co/lianghsun/privacy-filter-tw) 已发布繁体中文台湾场景模型。
本项目应以可验证的资产和协议形成差异化，而不是争夺宽泛的“首个”称号。

建议定位：

> 面向简体中文的 24 类 PII 开源可复现套件：同时发布人类/自然语境 surrogate 金标、
> 隐藏榜协议、model-only/rules-only/full-system 分榜、训练与评测代码以及证据绑定的模型包。

在数据和隐藏榜真正完成前，只能称为目标定位，不能写成已经实现的事实。

## 本项目当前可复现基线

下表使用同一 PII Bench ZH closed-8 gold、解码前标签约束和 exact `(start, end, label)`
strict character-span micro F1。不同赛道仍然分开，不能用 hybrid 分数暗示其中某个模型单独达到
相同水平。

| 对象 | 赛道 | 覆盖 | Formal | Chat | Pooled |
|---|---|---:|---:|---:|---:|
| 项目 v1 selected | `model_raw` | 8/8 | 0.5580 | 0.3890 | 0.4992 |
| 项目 no-class-weight seed 42 | `model_raw`（JPT 诊断） | 8/8 | 0.6090 | 0.4786 | 0.5669 |
| 中文 ERNIE NER | `model_raw` | 4/8 | 0.5790 | 0.5301 | 0.5631 |
| PII Engineer 中文 NER | `model_raw` | 4/8 | 0.7183 | **0.7073** | 0.7148 |
| PII Engineer Multi NER v2.1 | `model_raw` | 7/8 | 0.3897 | 0.3787 | 0.3860 |
| AIguard Qwen3-0.6B | `model_raw` | 8/8 | **0.8014** | 0.5714 | **0.7364** |
| OpenAI Privacy Filter | `model_raw` | 5/8 | 0.5840 | 0.4005 | 0.5306 |
| CLUENER2020 RoBERTa | `model_raw` | 2/8 | 0.5753 | 0.5735 | 0.5747 |
| OpenMed SnowflakeMed Large 568M | `model_raw` | 8/8 | 0.3832 | 0.4518 | 0.4045 |
| OpenMed QwenMed XLarge 600M | `model_raw` | 8/8 | 0.6552 | 0.5562 | 0.6239 |
| OpenMed BigMed Large 560M | `model_raw` | 8/8 | 0.3659 | 0.4362 | 0.3864 |
| OpenMed Privacy Filter Multilingual v2 | `model_raw` | 8/8 | 0.7229 | 0.5780 | 0.6775 |
| `cn_common_v5` | `rules_only` | 6/8 | 0.6780 | 0.6563 | 0.6710 |
| `argus-redact 0.7.18` fast | `rules_only` | 8/8 | **0.9355** | **0.9107** | **0.9275** |
| 历史 Presidio + ERNIE + `cn_common_v5` | `framework_hybrid`（custom fusion） | 8/8 | 0.9043 | 0.8786 | 0.8960 |
| 历史 Presidio + CLUENER + `cn_common_v5` | `framework_hybrid`（custom fusion） | 8/8 | 0.9527 | 0.9361 | 0.9474 |
| Presidio 2.2.363 native + ERNIE | `framework_hybrid`（release-native v2） | 8/8 | 0.90517850 | 0.87925378 | 0.89680396 |
| Presidio 2.2.363 native + CLUENER | `framework_hybrid`（release-native v2） | 8/8 | **0.95273279** | **0.93613793** | **0.94736372** |
| OpenMed SnowflakeMed hybrid | `framework_hybrid` | 8/8 | 0.7184 | 0.7997 | 0.7444 |
| OpenMed QwenMed hybrid | `framework_hybrid` | 8/8 | 0.7070 | 0.7476 | 0.7200 |
| OpenMed BigMed hybrid | `framework_hybrid` | 8/8 | 0.7515 | 0.8064 | 0.7687 |
| OpenMed Privacy Filter Multilingual hybrid | `framework_hybrid` | 8/8 | 0.7742 | 0.7262 | 0.7584 |

两组 Presidio 行必须按 runtime 分开解释。历史 `0.8960 / 0.9474` pooled 结果使用
`LocalRecognizer` 加外层 custom fusion；新 v2 是 release-native Presidio AnalyzerEngine
`2.2.363`，分开注册 NER 与 `CnCommonRecognizer`，由框架本身处理重复和 overlap，且全程
offline、CPU-only。v2 ERNIE pooled micro/macro 为 `0.89680396 / 0.83138344`
（TP/FP/FN `21578/3338/1628`），Formal/Chat macro 为 `0.84294878 / 0.78701966`；v2
CLUENER pooled micro/macro 为 `0.94736372 / 0.86673384`
（`21229/382/1977`），Formal/Chat macro 为 `0.87321090 / 0.83997016`。

Native v1 因 macro 验证的舍入顺序错误在 aggregate 发布前 fail closed，现已由
[failure receipt](../reports/evaluation/native_presidio_zh_ner_framework_hybrid_v1_failure_receipt.json)
标记 `WITHDRAWN_VALIDATION_FAILURE_SUPERSEDED`。修正后的 v2 有只读
[aggregate](../reports/baselines/native_presidio_zh_ner_framework_hybrid_v2.json)、
[result closure](../configs/evaluation/native_presidio_zh_ner_framework_hybrid_v2_result_closure.freeze.json)
和 [closure receipt](../reports/evaluation/native_presidio_zh_ner_framework_hybrid_v2_result_closure_receipt.json)，
但状态仍是 `RESULT_CLOSED_NON_ACTIVATING / BLOCKED_PUBLIC_TEST_EXPOSED`。PII Bench ZH 是
公开、已暴露、程序化合成数据，8,000 篇均含协议内 PII，无法估计 PII-free document FPR；两套
NER 权重许可也未闭合。因此权重再分发、release bundle、registry activation 和所有
first/best/SOTA claims 均禁用，不能依据这组结果继续调 cascade 路由。

另有一条与 PII Bench 完全不同的第一版服务回归硬门。用户提供的旧结果位于公开 CLUENER dev：
1,343 篇、10 类、3,072 个 span occurrence，旧完整服务 TP/FP/FN 为 `2525/820/547`，精确
micro F1 是 `5050/6417`。当前内部回归门 `docs/legacy_cluener_regression_gate.md` 为
`G0 PASS / G1 REPLAY_PASS_HISTORICAL_ATTRIBUTION_UNRESOLVED / G2-v2 FAIL / G2-v3 LIMITED_COMPATIBILITY_ELIGIBLE_NONACTIVATING`。
G1 逐行 replay 精确得到同一个 `5050/6417`，但 DOCX 未绑定其 commit/runtime。G2-v2 actual
service 得到 `2528/828/544`、`5056/6428`，严格门失败；在该结果披露后，阈值 `0.50` 的
G2-v3 actual service 得到 `2510/736/562`、`5020/6318`，严格超过 G1。[paired successor](../reports/evaluation/legacy_cluener_g1_g2_threshold_050_v3_comparison_v2.json)
重算 ΔF1=`+0.007583133655`、描述性 95% CI `[0.004883808071,0.010453811188]`，状态为
`CLOSED_LIMITED_COMPATIBILITY_REGRESSION_ELIGIBLE_NONACTIVATING`。单独 NER 仍只是 non-blocking
diagnostic。由于 v3 是 post-result/nonblind、public-dev/test-exposed，且 `book/game/movie/scene`
等标签不是 PII，`legacy_cluener_10` 只能作为非默认 compatibility profile；该结果不能充当生产
`pii_core`、hidden test 或最佳证据。证据来源保留在内部审计文档
`docs/first_version_baseline_audit.md`。

因此当前项目模型还没有通过 model-only 门槛；完整系统已经通过独立的 legacy CLUENER 兼容门，
但该合同与 closed-8 framework-hybrid 不兼容，也不关闭 selected/default、PII-free FPR、human
hidden、license、GPU/container 或 release 门。发布页面必须同时显示模型单体和系统分数；完整的 revision、映射、per-label 结果和
artifact hash 见 [`baseline_matrix.md`](baseline_matrix.md) 及 `reports/baselines/`。

argus-redact 的 fast mode 是确定性框架/规则轨，不加载 HanLP 或 LLM。本项目在固定
`0.7.18` wheel 与 source revision 上全量重跑，而不是引用其旧的 1,000-row 自报表。它与 PII
Bench ZH 同作者，framework 仓库又包含该数据的生成器和回归门禁，因此 `0.9275` 只能说明对
这个同源合成分布覆盖很强，不能支持独立榜单、自然语境或真实世界“最好”的主张。

OpenMed 四模型的八条 raw/hybrid 结果来自 v1 撤回后的 post-freeze implementation repair，且已在
全新目录完整执行；PII Engineer Multi v2.1 则先固定 revision、runner、ORT、mapping、默认阈值与
数据 manifest，再完成 8,000 条 CPU-only 复现。它的 Formal/Chat/Pooled micro F1 为
`0.38965434 / 0.37870855 / 0.38601245`，pooled macro 为 `0.29449612`，precision/recall 为
`0.76774028 / 0.25782125`：在当前 12 个同协议 raw 诊断对象中 pooled micro 排名 `12/12`。银行卡
不受支持、护照零命中，地址和车牌在 Chat 也均零命中，因此不能把它写成旧 PII Engineer Chinese
NER 的升级效果。两组 post-freeze 结果都来自已见、程序化合成且没有 PII-free 文档的 PII Bench；
它们不是 blind leaderboard，不可用于路由调参，也不能给出部署 document FPR。

OpenAI Privacy Filter 也已独立全量重跑，而不是用 pipeline argmax 代替其原生解码器。固定
revision、官方 constrained Viterbi/default operating point、softmax 前 closed-8 mask 下，
Formal/Chat/Pooled micro F1 为 `0.5840 / 0.4005 / 0.5306`，macro 为
`0.4290 / 0.2780 / 0.3905`。它只语义覆盖 5/8，且模型卡说明主要语言是英语；这组结果是有价值的
国际架构对照，但没有改变 AIguard/PII Engineer 构成的简中 model-only 包络。

CLUENER 的裸模型只支持姓名/地址；历史 custom-fusion 和 release-native v2 的高 hybrid 分数都
不能归因给单个 checkpoint，也不能互相混写 runtime。固定模型卡未声明许可证，原始 CLUENER
数据又来自 THUCTC/Sina News RSS，因此本项目不再分发其权重、不以其为公开权重训练起点，
只把固定 hash 结果作为工程框架比较。

首轮 AIguard 字符边界迁移 pilot 在冻结后仅运行过一次外部评测，得到 Formal/Chat/Pooled micro
F1 `0.7726 / 0.6756 / 0.7426`。它虽略高于 AIguard 的 pooled 值，却仍分别低于 Formal 与 Chat
逐套件门槛，因此没有通过“最佳”条件。该结果已标为 `test_already_seen` 和
`not_release_candidate`；后续数据或模型选择不得再使用 PII Bench。主要聚合短板为地址 exact
边界，以及姓名、电话和车牌召回；优化必须改用独立 validation 和未查看的 v2 hidden test。

一次在 holdout 已经可见后进行的 model-only 阈值诊断，把 synthetic/formal/chat strict F1
分别提高到 0.6601/0.5348/0.4345，但召回明显下降。它只用于定位问题，不是事前冻结的发布结果，
不得用于模型选择或 headline。新的阈值和解码器必须只在 v2 validation 上拟合并在 hidden test
前冻结。

## 可利用的公开比较

| 资源 | 数据/榜单事实 | 可比性 | 本项目用途 |
|---|---|---|---|
| [DataFountain 472：非结构化商业文本隐私识别](https://www.datafountain.cn/competitions/472) | 官方称数据源于真实商业交互并已脱敏、标注；正式赛结束后保留经典赛提交排行榜 | 最接近真实中文商业语境，但标签包含商业秘密，格式、许可和评测口径与本项目不同；页面没有开放数据许可证 | 只按平台协议做外部 blind submission；不得把下载数据混入训练、HF 数据集或本项目 public test |
| [PII Bench ZH](https://huggingface.co/datasets/wan9yu/pii-bench-zh) | formal 5,000 + chat 3,000，8 类，精确字符 offsets，Apache-2.0；官方声明 100% 程序化合成 | 可复现但不是真实语境；截至本快照 HF 官方 leaderboard API 返回 0 个 entry | 保持 evaluation-only；建立 `closed-8` 和 `open-24` 两种固定协议，不再用于模型选择 |
| [tw-PII-bench](https://huggingface.co/datasets/lianghsun/tw-PII-bench) | 910 条繁中、8 个 in-schema + 11 个台湾 OOD + hard negatives，Apache-2.0；数据卡声明 PII 为虚构 | 语言区域、标签和 OOD 映射不同；没有 HF 官方榜 entry | 做繁中迁移诊断，不纳入简中主榜 |
| [privacy-filter-tw](https://huggingface.co/lianghsun/privacy-filter-tw) | 作者在 tw-PII-bench 上自报全体 in-schema strict micro F1 0.727，base 为 0.541；训练集为 183,588 条合成数据 | 同 benchmark 内可参考；不是独立评测，且 19 类、OPF Viterbi 与本项目 24 类 Qwen 解码不同 | 复现其公开 evaluator 后建立独立 cross-locale baseline；明确标记 self-reported/independently reproduced |
| [PII Engineer benchmark](https://pii.engineer/benchmarks) | 作者自报 1,200 条、9 语言、9 类；中文 F1 0.918；本项目固定其公开 revision 后独立得到 closed-8 pooled 0.7148 | 自报 gold/split 未开放为标准隐藏榜；公开 artifact 实际为 BERT BIO ONNX，且模型卡架构与许可证声明冲突 | 保留独立报告但禁止再分发权重；自报 0.918 不进入同协议排序 |
| [PII Engineer Multi NER v2.1](https://huggingface.co/pii-engineer/PII-Engineer-Multi-NER-v2.1) | 多语言 ONNX span model；本项目固定 `8ce12a2…f244` 后独立得到 closed-8 pooled micro/macro `0.3860 / 0.2945` | HF frontmatter Apache-2.0、正文 AGPL-3.0、GitHub 源码 Apache-2.0 尚未调和；public-test-exposed synthetic，PII-free FPR 不可测 | 只作为不可再分发、non-activating 技术对照；未来仅在独立 human dev 验证条件分支的 unique TP |
| [AIguard-pii-detection-fast](https://huggingface.co/ZJUICSR/AIguard-pii-detection-fast) | Qwen3-0.6B、21 类，作者自报 entity F1 0.9629；本项目固定 revision 后独立得到 closed-8 pooled 0.7364 | 作者测试集与统一 evaluator未开放；其自报 0.9629 仍不可直接比较 | 撤回 generation-1 的 independently reproduced 诊断值；不是活跃门槛，等待独立 human hidden successor |
| [OpenMed Chinese PII family（含 QwenMed XLarge 600M）](https://huggingface.co/OpenMed/OpenMed-PII-Chinese-QwenMed-XLarge-600M-v1) | 本项目对 SnowflakeMed/QwenMed/BigMed/Privacy Filter Multilingual 固定 revision 后完成 raw+hybrid `8/8`；QwenMed raw pooled `0.6239`，四路 raw 最佳为 Multilingual `0.6775` | post-freeze repair、已见合成 public test；hybrid 与 raw 是不同赛道；无 PII-free FPR、D1 或 human hidden | 保留描述性同协议对照；不得把 hybrid 增益归因给权重，亦不构成“最强”证据 |
| [GLiNER2-PII](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi) | 0.3B、42 类、Apache-2.0；作者在 SPY 自报 exact span F1 0.477 | 训练的 7 种语言均为欧洲语言，模型卡明确称非欧洲 locale/script 尚未评测；不是中文专项 | 作为国际架构背景，不列入当前中文主表；后续可做 zero-shot 诊断 |
| [OpenAI Privacy Filter](https://github.com/openai/privacy-filter) | 8 类、1.5B total/50M active、原生 constrained Viterbi、Apache-2.0，主要语言为英语；本项目独立得到 closed-8 pooled micro/macro `0.5306 / 0.3905` | 只语义覆盖本协议 5/8；不是中文专项，身份证/护照/车牌计 FN | 保留固定 revision、官方 decoder/default OP 的国际架构基线；结果不进入模型选择 |
| [PIIBench](https://huggingface.co/datasets/Pritesh-2711/pii-bench) | 统一 10 个公开来源、48 类，HF 当前提供 80/10/10 的 1M 行版本；论文报告 8 个公开系统的 span F1 均低于 0.14 | HF 数据卡语言标为 English，且把来源重新切分，不能作为简中 hidden test；也存在训练来源重叠风险 | 作为国际开放协议和 evaluator 设计参考；做 overlap/provenance 审计后才能加入跨语言外测 |
| [MultiPriv](https://huggingface.co/datasets/CyberChangAn/MultiPriv) | 中英文、多模态隐私研究集；数据卡称文本为合成、脱敏或合法来源，并采用 CC BY-NC-SA 4.0 | 来源类型混合、非商业/ShareAlike 限制，且主要面向隐私推理而非统一 exact-span 中文 NER 排行 | 只列 research-only 外部诊断候选；不能进入目标 Apache 权重训练或替代人类语境 gold |

第三方网页上的数字统一标注为 `self-reported`，直到本项目使用冻结 revision、公开 evaluator、
一致标签映射和相同 gold 独立复现。不同资源之间不制作“总榜第一”的合并排序。

## v2 的统一五赛道协议

每个 benchmark 固定五条互不混用的赛道：

1. `model_raw`：checkpoint + 随发布物冻结的确定性解码，不拟合 benchmark 阈值；
2. `model_calibrated`：只允许在公开或私有 dev 上拟合温度、阈值或 transition，完全不使用规则；
3. `rules_only`：版本化规则和 validator，不调用模型；
4. `framework_hybrid`：固定开源框架 + 指定公开 NER/model + 版本化规则；
5. `full_system`：模型、规则、融合、校准和后处理完整合同。

旧聚合报告中的 `model_only` 是 `model_raw` 的兼容 alias；新 artifact 只写 canonical enum。

PII Bench ZH 另外分为：

- `closed-8`：推理前冻结 8 类 label mask，只比较共同标签；
- `open-24`：允许模型输出 24 类，所有不在 gold 中的额外输出照常计 FP。

主指标固定为 strict character-span micro F1，同时报告：

- strict macro F1、precision、recall、每类支持量和每类 P/R/F1；
- PII-free document FPR、FP/1K characters；
- start/end boundary accuracy、fragmentation rate、character F1；
- calibration ECE/Brier；
- 1,000 次 document bootstrap 95% CI；
- 固定硬件上的 throughput、P50/P95 latency 和 peak GPU memory。

Relaxed F1 和 character F1 只能做边界诊断，不能取代 strict span headline。阈值、label map、
decoder 和 evaluator revision 必须在 hidden test 提交前冻结并记录 SHA-256。

## 社区榜单建议

公开 `public_test` 只负责复现；长期榜单使用 2,000 篇不上传文本和 gold 的
`hidden_leaderboard`，并至少包含 human-authored context、surrogate PII、PII-free、hard negative、
长文档和 domain/source/time holdout。

提交物只包含预测 span 和模型/系统声明，不接受可执行任意代码。服务器端执行统一 evaluator，
返回聚合分数和有限错误桶，限制每日提交次数，定期轮换 hidden 子集。排行榜分别显示 model-only
和 full-system，不允许未声明规则、外部 API 或人工修正的提交进入 model-only 轨。

只有当真实语境数据的许可、隐私、双标/仲裁和 hidden evaluator 全部落地后，这套榜单才构成
比纯合成自报分数更有说服力的社区贡献。
