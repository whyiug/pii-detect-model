# release-eval-v2 校准与 internal 解锁流程

这套门禁把三个结果轨道严格分开：

- `model_raw`：选中模型、阈值 0、未应用 calibration；
- `model_calibrated`：同一模型和冻结 calibration，社区 builder 的 `model-only` 模式，规则分支关闭；
- `full_system`：同一模型和 calibration，社区 builder 的 `cascade` 模式，规则/模型级联开启。

正式 v2 的 `calibration` 只允许第一种。`internal_evaluation` 在模型、calibration、服务配置和
实现源码全部冻结前，不得 open、hash、解析或打印。以下命令中的 `<RUN>` 必须是新目录；所有
receipt/binding 都是 no-clobber、只读、path-free 的聚合 JSON。

## 1. 生成 calibration 授权

先完成三 seed 选择，并生成/严格重放 v1→v2 amendment。然后执行：

```bash
python scripts/release_eval_v2_stage_gate.py build-calibration-authorization \
  --protocol configs/train/aiguard24_sota_v3_resplit_selection.json \
  --development-manifest <V2_RESPLIT>/dataset_manifest.json \
  --v1-release-manifest <RELEASE_EVAL_V1>/dataset_manifest.json \
  --candidate <SEED13_MODEL> --candidate <SEED42_MODEL> --candidate <SEED97_MODEL> \
  --selection-receipt <RUN>/aiguard24_v3_selection.json \
  --amendment <RUN>/aiguard24_v3_release_eval_v2_amendment.json \
  --dataset-manifest <RELEASE_EVAL_V2>/dataset_manifest.json \
  --materialization-receipt <RELEASE_EVAL_V2>/supersession_receipt.json \
  --freeze-receipt configs/evaluation/synthetic_sota_release_eval_v2_supersession_freeze.json \
  --calibration-gold <RELEASE_EVAL_V2>/calibration.jsonl \
  --model-artifact <SELECTED_MODEL> \
  --output <RUN>/calibration.authorization.json
```

该入口先重放冻结 selector 和 amendment，确认选中训练 manifest 的 logical SHA、权重、模型
artifact 与 v2 固定身份一致，之后才允许读取 calibration；它不会打开 internal。

## 2. 生成 calibration `model_raw`

```bash
python scripts/generate_release_eval_v2_candidate_predictions.py \
  --mode model-raw --target-split calibration --scope open24 \
  --model <SELECTED_MODEL> \
  --input <RELEASE_EVAL_V2>/calibration.jsonl \
  --calibration-authorization <RUN>/calibration.authorization.json \
  --predictions <RUN>/calibration.model_raw.jsonl \
  --receipt <RUN>/calibration.model_raw.generation.json \
  --device cuda:0 --document-batch-size 16 --micro-batch-size 8
```

GPU 执行前仍须先检查 `nvidia-smi`，只使用空闲卡并通过 `CUDA_VISIBLE_DEVICES=<单卡>` 隔离。
生成器在推理前验证门禁，随后只对一个稳定的私有 snapshot 做推理与哈希，避免输入 A→B TOCTOU。

接着为 calibration raw prediction 建立并严格 replay provenance。`build` 和 `replay` 使用相同参数：

```text
--track model_raw --target-split calibration
--predictions <RUN>/calibration.model_raw.jsonl
--generation-receipt <RUN>/calibration.model_raw.generation.json
--gold <RELEASE_EVAL_V2>/calibration.jsonl
--dataset-manifest <RELEASE_EVAL_V2>/dataset_manifest.json
--materialization-receipt <RELEASE_EVAL_V2>/supersession_receipt.json
--freeze-receipt configs/evaluation/synthetic_sota_release_eval_v2_supersession_freeze.json
--model-artifact <SELECTED_MODEL>
--selection-receipt <RUN>/aiguard24_v3_selection.json
--amendment <RUN>/aiguard24_v3_release_eval_v2_amendment.json
--protocol configs/train/aiguard24_sota_v3_resplit_selection.json
--development-manifest <V2_RESPLIT>/dataset_manifest.json
--v1-release-manifest <RELEASE_EVAL_V1>/dataset_manifest.json
--candidate <SEED13_MODEL> --candidate <SEED42_MODEL> --candidate <SEED97_MODEL>
```

## 3. 拟合并冻结 calibration

`--model-version` 必须是选中 `training_manifest.json` 的 `manifest_sha256`，不是目录名：

```bash
python scripts/calibrate_predictions.py \
  --gold <RELEASE_EVAL_V2>/calibration.jsonl \
  --predictions <RUN>/calibration.model_raw.jsonl \
  --fit-prediction-manifest <RUN>/calibration.model_raw.provenance.json \
  --release-eval-v2-authorization <RUN>/calibration.authorization.json \
  --dataset-manifest <RELEASE_EVAL_V2>/dataset_manifest.json \
  --expected-dataset-manifest-sha256 bd99b9a858c07fe96f1effcc93d07defa17f7ce39a00712bcfe6b9fda56c19ad \
  --supersession-freeze configs/evaluation/synthetic_sota_release_eval_v2_supersession_freeze.json \
  --expected-supersession-freeze-sha256 06ebdce5e288b0e7699a8225e28c4c4f0a1762e76f6967cf990b14384101f0f0 \
  --model-version <SELECTED_TRAINING_MANIFEST_SHA256> \
  --output <RUN>/calibration.bundle.json \
  --diagnostics <RUN>/calibration.diagnostics.json \
  --t0-recall-floor 0.90 --t1-recall-floor 0.80 --fit-temperature
```

正式 seed97 链固定使用 T1 recall floor `0.80`。此前 `0.85` 的 calibration 尝试因
`DEVICE_ID` 的最大候选召回约为 `0.815445` 而无法满足约束；把 floor 调整为 `0.80` 只属于
calibration 阶段的可达性修正，没有重选 seed、模型或 checkpoint。最终 r3 服务 diagnostic 继续使用
同一冻结 calibration bundle；仅较早的 r2 服务 diagnostic 被 supersede。

## 4. 最终冻结并解锁 internal

```bash
python scripts/release_eval_v2_stage_gate.py build-internal-unlock \
  <与第1步相同的 selector/amendment/dataset/model 参数> \
  --authorization <RUN>/calibration.authorization.json \
  --calibration-bundle <RUN>/calibration.bundle.json \
  --calibration-diagnostics <RUN>/calibration.diagnostics.json \
  --calibration-fit-predictions <RUN>/calibration.model_raw.jsonl \
  --calibration-fit-generation-receipt <RUN>/calibration.model_raw.generation.json \
  --calibration-fit-prediction-manifest <RUN>/calibration.model_raw.provenance.json \
  --model-binding-output <RUN>/final-model-binding.v2.json \
  --service-binding-output <RUN>/service-configuration-binding.v2.json \
  --unlock-output <RUN>/internal.unlock.json
```

该命令机械生成与 quality v2 兼容的：

- `pii-zh.community-final-model-binding.v2`；
- `pii-zh.community-service-configuration-binding.v2`；
- `pii-zh.release-eval-v2-internal-unlock.v1`。

第一次 internal 访问前，必须用 `replay-internal-unlock` 和同一组输入严格重放成功。对应 schema：
`configs/evaluation/release_eval_v2_calibration_authorization_v1.schema.json` 和
`configs/evaluation/release_eval_v2_internal_unlock_v1.schema.json`。

当前正式链已经完成该步骤：`internal.unlock.json` 状态为
`INTERNAL_EVALUATION_AUTHORIZED_ONE_SHOT`，strict replay PASS；随后三条 internal prediction
provenance 也全部 strict replay PASS。internal 已经一次性查看，故同一候选的模型、calibration、
规则、validator、路由、fusion 与服务配置从此不可回改。

## 5. 一次性生成 internal 三轨

三次生成都必须带以下门禁参数：

```text
--target-split internal_evaluation
--internal-unlock <RUN>/internal.unlock.json
--final-model-binding <RUN>/final-model-binding.v2.json
--service-configuration-binding <RUN>/service-configuration-binding.v2.json
```

具体模式：

| 结果轨 | generator `--mode` | calibration 参数 | provenance `--track` |
|---|---|---|---|
| 阈值0模型 | `model-raw` | 禁止 | `model_raw` |
| 校准后单模型 | `model-only` | `--calibration <RUN>/calibration.bundle.json` | `model_calibrated` |
| 最终级联服务 | `cascade` | `--calibration <RUN>/calibration.bundle.json` | `full_system` |

`model_calibrated` 和 `full_system` 的 provenance 还必须通过三个 `--model-raw-*` 参数绑定同 split
的 raw predictions、generation receipt 和 provenance manifest。三轨都复用同一个选中模型、bundle、
selector/amendment 证据。看到 internal 结果后，不得改模型、阈值、calibration、路由、校验器或规则。

quality producer 和 registry 的正式 v2 调用还必须传入：

```text
--internal-unlock <RUN>/internal.unlock.json
--final-model-binding <RUN>/final-model-binding.v2.json
--service-configuration-binding <RUN>/service-configuration-binding.v2.json
```

它们会在读取 internal gold 前调用同一个 pre-open guard。
