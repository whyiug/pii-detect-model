# GitHub / Hugging Face 发布手册（community v2 RC1）

> 本文件是操作模板，不是执行日志。2026-07-19 本地快照中没有执行任何远端创建、push、tag
> 上传、GitHub Release 或 Hugging Face 写入。每条远端写命令都必须在对应 checkbox 完成、
> 维护者再次确认目标和可见性后逐条执行。

## 1. 当前快照与硬停止条件

- [x] 本地冻结回执存在于
  `$RELEASE_RUN/community-cascade-release-v2.receipt.v2r3.json`。
- [x] 回执状态为 `READY_FOR_USER_AUTHORIZATION`，回执内 `receipt_sha256` 为
  `087b1c0e1b918e2c927c0653f312d8d0dc2954b6631b46fa22f9d7e194a84880`，回执文件
  SHA-256 为 `5248a9746ce0676856cb29ecdd183f232a6078cd4b94e7138c539fa5bc1675e7`。
- [x] GitHub CLI 已登录。
- [ ] GitHub 目标 `whyiug/pii-detect-model` 尚不存在；当前本地仓库也没有 remote。
- [x] Hugging Face CLI 已在官方 Hub 验证登录，当前用户为 `Forrest20231206`，可见组织为
  `SparkShieldLab`；本文件没有记录或输出 token。
- [ ] 发布 namespace 尚未在 `Forrest20231206` 与 `SparkShieldLab` 之间确定；对两个候选 repo id 的
  只读查询均返回不存在或不可访问，未创建任何仓库。
- [ ] human/legal 许可证审批未完成；license inventory 状态仍为
  `COMPLETE_HUMAN_APPROVAL_PENDING`。
- [ ] `$RELEASE_RUN/model-package-v2r2` 不是可上传包：其中 `SECURITY.md` 是旧版且明确说没有
  已测试私密报告渠道，模型卡仍为 `unpublished_local_candidate`，并含
  `community_v2_preauthorization.json`。其中 false-only config/remote-code 字段是冻结候选 lineage，
  publication-successor 会原样保留并由外部批准回执解释，不能把它本身当成发布授权或手工翻转。
- [ ] `$RELEASE_RUN/dist-v2r3` 中的 wheel 早于当前 source release-preparation commit，只能作为历史
  验收证据；GitHub Release 必须从最终 source commit 在新目录重建 wheel，不能复用旧文件。

以上未完成项全部是硬停止条件。尤其不能把本地 `READY_FOR_USER_AUTHORIZATION` 解释为远端发布
授权，不能为了赶发布直接上传 `model-package-v2r2`。

## 2. 固定变量

下面只定义占位符。不要把 token 写入命令、脚本、日志或本文件。

```bash
export RELEASE_RUN="<local-frozen-release-run>"
export PUBLICATION_DIR="$RELEASE_RUN/model-package-publication-successor-v1"
export DIST_DIR="$RELEASE_RUN/dist-publication-successor-v1"
export GH_REPO="whyiug/pii-detect-model"
export RELEASE_BRANCH="agent/community-v2-release-prep"
export RELEASE_TAG="v0.2.0rc1"
export HF_NAMESPACE="<namespace>"
export HF_REPO_ID="$HF_NAMESPACE/pii-zh-qwen3-0.6b-24class"
export PUBLICATION_RECEIPT="$RELEASE_RUN/publication-receipt.v1.json"
export PUBLICATION_MODEL_CARD="$RELEASE_RUN/README.publication.final.md"
export LICENSE_APPROVAL_RECEIPT="$RELEASE_RUN/human-license-approval-receipt.json"
export SECURITY_CHANNEL_RECEIPT="$RELEASE_RUN/tested-private-security-channel-receipt.json"
```

推荐的 Hugging Face repo id 是
`<namespace>/pii-zh-qwen3-0.6b-24class`。`<namespace>` 必须以重新登录后的
`hf auth whoami` 结果为准，不得猜测或代填。

## 3. 总体验收清单

### 3.1 本地源码与版本

- [ ] 发布 commit 只包含经审核的源码、配置、测试、文档和发布元数据，没有运行目录、缓存、
  模型权重、token、真实 PII 或与本次发布无关的工作树改动。
- [ ] 最终 wheel 由最终 source commit 的干净 checkout 构建，metadata 为 `0.2.0rc1`；Git tag 为
  `v0.2.0rc1`，当前模型卡、
  Python 包、容器标签、release notes 与服务响应没有旧 `synthetic-v1.3-rc1` 或冲突版本。
- [ ] current model card 是 publication-successor 中重新生成并复核的版本；它不再把候选写成
  `unpublished_local_candidate`，也不预写尚未发生的上传成功状态。
- [ ] GitHub 源码 commit 已冻结并记录完整 40 位 SHA。
- [ ] 使用可验证签名创建 source tag；如果签名配置不可用，则停止，不静默降级为未签名 tag。
- [ ] 本地 unit/release 测试、wheel smoke、离线模型/服务 smoke、容器 smoke、SBOM、checksums、
  public artifact scan 均通过并绑定同一候选。
- [ ] GitHub hosted CI 在最终 source commit 上通过；本地测试不能替代 hosted CI。

### 3.2 安全与许可证

- [ ] human/legal 已逐项审核许可证清单，审批人、时间、结论和例外均有留痕；不能只依赖自动
  license scanner 或上游模型卡中的许可证字段。
- [ ] GitHub 私密漏洞报告已启用，并由维护者使用不含真实 PII/secret 的最小测试报告验证。
- [ ] 根 `SECURITY.md` 与 publication-successor 内 `SECURITY.md` 都指向已实测的私密渠道，
  不再包含“No tested private reporting route exists”或“Public upload is withheld”。
- [ ] 发布扫描确认源码、wheel、模型包、release notes、SBOM 和回执不含 secret、凭据、真实 PII、
  本机绝对用户名路径或冻结逐行评测数据。

### 3.3 publication-successor

- [ ] 保留 `$RELEASE_RUN/model-package-v2r2` 为只读证据，不原地修改、不上传。
- [ ] 由经过审核的生成流程创建新的 `$PUBLICATION_DIR`；模型和 tokenizer 身份必须与冻结候选一致，
  任何允许变化的发布元数据都列入 successor manifest。
- [ ] successor 不把 `community_v2_preauthorization.json` 当作发布授权，也不保留
  `pending_community_contract` 或 `unpublished_local_candidate` 发布状态。冻结 config、training
  manifest 与 remote code 的 false-only candidate-lineage 合同保持原字节；manifest 必须明确其
  不代表当前远端状态，发布授权只来自已校验的外部回执。
- [ ] current model card 只陈述具名 synthetic Open24 结果，并完整披露 100% 合成数据、PII Bench
  公开测试已见/posthoc/model_raw-only/full-service-N/A 限制。
- [ ] current model card 和 release notes 禁止使用“首个”、`first`、`SOTA`、“最好”、“最强”、
  “行业领先”或 `production-ready`，也不作合规保证。
- [ ] successor 的 `LICENSE`、`NOTICE`、`THIRD_PARTY_NOTICES.md` 与 human/legal 结论一致。
- [ ] successor 重新生成 `checksums.txt`，并通过离线加载、有限前向、remote-code 审计、文件闭包和
  public artifact scan。
- [ ] `$PUBLICATION_DIR` 中没有 `wheelhouse/` 和 `.whl`；HF 上传源只能是 `$PUBLICATION_DIR`，
  不能误指向 `$RELEASE_RUN`。

### 3.4 远端身份与双向绑定

- [ ] GitHub target 是精确的 `whyiug/pii-detect-model`，remote URL 经人工复核且不含凭据。
- [ ] GitHub source commit 与签名 tag 已推送；tag 验证通过且指向 hosted CI 通过的最终 commit。
- [ ] Hugging Face target 是精确的
  `<namespace>/pii-zh-qwen3-0.6b-24class`，且 token 只授予所需最小权限。
- [ ] Hugging Face 上传完成后记录不可变的完整 commit SHA；不以 `main` 这种可移动引用代替。
- [ ] HF current model card 在上传前已写入 GitHub repo、完整 source commit、签名 tag 和确定的
  GitHub Release URL，形成 HF → GitHub 绑定。
- [ ] 最终 `publication-receipt.json` 记录 GitHub repo、完整 source commit、签名 tag、HF repo id、
  HF immutable commit、模型/模型卡/checksums SHA-256 和发布时间，形成 GitHub → HF 绑定。
- [ ] `publication-receipt.json` 经 schema/哈希校验后附到 GitHub Release；GitHub Release notes
  指向 HF 的 immutable commit，而不是只指向 `main`。
- [ ] 双向链接逐项打开验证；发布回执不得通过自引用或手抄 SHA 伪造闭包。

## 4. 命令模板：只在全部硬门关闭后执行

以下命令本轮**没有执行**。执行者必须逐段运行并检查输出，禁止整段无审阅粘贴。

### 4.1 只读确认与本地冻结

```bash
git status --short
git diff --check
git rev-parse --verify HEAD
gh auth status
hf auth whoami
sha256sum "$RELEASE_RUN/community-cascade-release-v2.receipt.v2r3.json"
```

如果 Hugging Face 登录在执行时失效，只使用交互式登录，不能把 token 放在命令行参数、环境导出
或 shell history 中：

```bash
# hf-mirror 只用于国内下载；创建仓库和上传必须指向官方 Hub。
export HF_ENDPOINT=https://huggingface.co
hf auth login
hf auth whoami
```

登录后确认 `HF_NAMESPACE`，再重新设置并打印 repo id；不要打印 token。

```bash
export HF_NAMESPACE="<namespace-from-whoami>"
export HF_REPO_ID="$HF_NAMESPACE/pii-zh-qwen3-0.6b-24class"
printf '%s\n' "$HF_REPO_ID"
```

### 4.2 建立私有 GitHub 暂存仓库和安全渠道

先以 private 创建，建立并实测私密安全渠道；本轮不执行：

```bash
gh repo create "$GH_REPO" --private --description "Simplified-Chinese PII model and cascade research preview"
git remote add origin "https://github.com/${GH_REPO}.git"
git remote -v
git push --set-upstream origin "$RELEASE_BRANCH"
gh api --method PUT "repos/${GH_REPO}/private-vulnerability-reporting"
gh api "repos/${GH_REPO}/private-vulnerability-reporting"
```

随后由维护者用合成内容实测私密报告入口，更新并复核两个 `SECURITY.md`。仅检查 API 返回
“enabled”还不算完成实测。

### 4.3 冻结 source commit 与签名 tag

在安全渠道、许可证和最终源码都关闭后，记录 commit 并签名；本轮不执行：

```bash
export SOURCE_COMMIT="$(git rev-parse HEAD)"
git tag -s "$RELEASE_TAG" "$SOURCE_COMMIT" -m "pii-zh community v2 RC1"
git tag -v "$RELEASE_TAG"
git push origin "$RELEASE_BRANCH"
gh run list --repo "$GH_REPO" --branch "$RELEASE_BRANCH" --limit 10
```

只有最终 source commit 的 GitHub hosted CI 全部通过并经人工核对后，才单独执行下面的 tag push；
本轮不执行：

```bash
git push origin "$RELEASE_TAG"
```

若在此后修改任何版本、
模型卡、`SECURITY.md` 或发布元数据，必须重新冻结 commit、重跑 CI 并重建 tag；不得移动已公开 tag。

从最终 commit 的干净 checkout 构建新的 wheel 输出；目录必须不存在，且只能产生一个 wheel：

```bash
test ! -e "$DIST_DIR"
mkdir "$DIST_DIR"
uv build --wheel --out-dir "$DIST_DIR"
test "$(find "$DIST_DIR" -maxdepth 1 -type f -name '*.whl' | wc -l)" -eq 1
```

随后重跑 wheel inventory、隔离安装、CLI/service smoke、SBOM、public scan 和哈希记录。上面的构建
不能替代 GitHub hosted package-smoke。

### 4.4 publication-successor 预检

publication-successor 必须由受审 builder 生成；不能用 `cp` 后手改几行冒充重新验收。先把已审
模型卡模板中的唯一 HF namespace 占位符渲染为登录后确认的精确 repo id。渲染文件使用 no-clobber，
且不写 token：

```bash
python - <<'PY'
import os
from pathlib import Path

source = Path("model_cards/PII_ZH_QWEN3_0_6B_24CLASS_PUBLICATION.md")
target = Path(os.environ["PUBLICATION_MODEL_CARD"])
placeholder = "HF_NAMESPACE/pii-zh-qwen3-0.6b-24class"
text = source.read_text(encoding="utf-8")
if text.count(placeholder) != 1:
    raise SystemExit("publication Model Card must contain exactly one HF target placeholder")
text = text.replace(placeholder, os.environ["HF_REPO_ID"])
if os.environ["GH_REPO"] not in text or os.environ["HF_REPO_ID"] not in text:
    raise SystemExit("publication Model Card target binding failed")
target.parent.mkdir(parents=True, exist_ok=True)
with target.open("x", encoding="utf-8", newline="\n") as handle:
    handle.write(text)
PY
```

两个人工回执必须满足
`configs/release/community_v2_publication_successor.schema.json` 中的封闭定义并校验自哈希；不得由
调用者自报 `PASS`。在最终源码 commit、许可证人工审批和私密安全渠道实测都完成后，先只读
preflight，再用完全相同参数构建：

```bash
COMMON_SUCCESSOR_ARGS=(
  --source-package "$RELEASE_RUN/model-package-v2r2"
  --output "$PUBLICATION_DIR"
  --model-card "$PUBLICATION_MODEL_CARD"
  --security SECURITY.md
  --notice NOTICE
  --third-party-notices THIRD_PARTY_NOTICES.md
  --final-local-receipt "$RELEASE_RUN/community-cascade-release-v2.receipt.v2r3.json"
  --human-license-approval-receipt "$LICENSE_APPROVAL_RECEIPT"
  --tested-private-security-channel-receipt "$SECURITY_CHANNEL_RECEIPT"
  --git-source-commit "$SOURCE_COMMIT"
  --github-repository "$GH_REPO"
  --hugging-face-repository "$HF_REPO_ID"
)
python scripts/build_community_v2_publication_successor.py "${COMMON_SUCCESSOR_ARGS[@]}" --dry-run
python scripts/build_community_v2_publication_successor.py "${COMMON_SUCCESSOR_ARGS[@]}"
```

若审批、target、哈希、源包闭包、false-only lineage 或保留元数据扫描任一不成立，builder 必须输出
`status=BLOCKED`、退出码 2，并且不创建 `$PUBLICATION_DIR`。生成后至少执行以下不写远端的检查：

```bash
test -d "$PUBLICATION_DIR"
test -f "$PUBLICATION_DIR/README.md"
test -f "$PUBLICATION_DIR/SECURITY.md"
test -f "$PUBLICATION_DIR/checksums.txt"
test ! -e "$PUBLICATION_DIR/community_v2_preauthorization.json"
test ! -d "$PUBLICATION_DIR/wheelhouse"
test -z "$(find "$PUBLICATION_DIR" -maxdepth 1 -type f -name '*.whl' -print -quit)"
! rg -n 'unpublished_local_candidate|pending_community_contract|No tested private reporting route exists|Public upload is therefore withheld' \
  "$PUBLICATION_DIR/README.md" "$PUBLICATION_DIR/SECURITY.md"
(cd "$PUBLICATION_DIR" && sha256sum --check checksums.txt)
```

还必须运行项目正式的 successor schema、离线模型加载、有限前向、安全扫描与 public artifact scan；
不能用上面几条文本检查替代正式 gate。

### 4.5 上传到私有 Hugging Face repo 并冻结 commit

只上传 `$PUBLICATION_DIR`。不要上传 `$RELEASE_RUN/wheelhouse/`、整个 `$RELEASE_RUN`、本地缓存或
GitHub 源码工作树；本轮不执行：

```bash
hf repo create "$HF_REPO_ID" --repo-type model --private --exist-ok
hf upload "$HF_REPO_ID" "$PUBLICATION_DIR" . \
  --repo-type model \
  --commit-message "Release pii-zh-qwen3-0.6b-24class v0.2.0rc1"
```

用只读 API 取得不可变 commit；命令不输出凭据：

```bash
export HF_COMMIT="$(python - <<'PY'
import os
from huggingface_hub import HfApi

info = HfApi().repo_info(os.environ['HF_REPO_ID'], repo_type='model')
print(info.sha)
PY
)"
printf '%s\n' "$HF_COMMIT"
hf repo tag create "$HF_REPO_ID" "$RELEASE_TAG" \
  --repo-type model \
  --revision "$HF_COMMIT" \
  --message "pii-zh community v2 RC1"
```

下载该 immutable commit 到新的临时目录，重跑 checksum、离线加载与最小前向；不要只验证本地上传
源目录。

### 4.6 生成 publication receipt、GitHub prerelease 与公开切换

最终 publication receipt 必须由受审脚本从实际远端读取 GitHub commit/tag 验证结果与
`HF_COMMIT` 后生成，禁止手写 JSON 或复制示例值。验证通过后，GitHub Release 只附带最终 wheel、
SBOM、checksums 和 receipt；明确不附带 wheelhouse。本轮不执行：

```bash
gh release create "$RELEASE_TAG" \
  --repo "$GH_REPO" \
  --title "pii-zh community v2 RC1" \
  --notes-file release/community-v2-rc1/RELEASE_NOTES.md \
  --prerelease \
  --draft \
  "$DIST_DIR/pii_zh_qwen-0.2.0rc1-py3-none-any.whl" \
  "$PUBLICATION_RECEIPT"
```

将 SBOM/checksums 加入上面的资产列表前，先把各自经过验收的精确路径赋给变量；不要使用
`$RELEASE_RUN/**` 或 shell 宽泛通配。人工检查 GitHub draft、HF private repo、私密报告入口、
双向链接和 immutable revision 后，才可按平台界面/受审 API 将两个仓库切换为 public 并发布
GitHub prerelease。可见性切换是最后一道单独授权的远端写操作，不应夹在上传脚本中自动执行。

确认获得这道单独授权后，下面是精确的最后切换模板；本轮不执行：

```bash
gh api --method PATCH "repos/${GH_REPO}" -f visibility=public

python - <<'PY'
import os
from huggingface_hub import HfApi

HfApi().update_repo_settings(
    repo_id=os.environ['HF_REPO_ID'],
    repo_type='model',
    private=False,
)
PY

export GITHUB_RELEASE_ID="$(
  gh api "repos/${GH_REPO}/releases/tags/${RELEASE_TAG}" --jq '.id'
)"
gh api --method PATCH \
  "repos/${GH_REPO}/releases/${GITHUB_RELEASE_ID}" \
  -F draft=false \
  -F prerelease=true
```

三步之间逐项执行只读检查；任何一步返回的 repo、commit、tag、可见性或 release id 不符合预期，
立即停止后续写入，不覆盖远端对象。

## 5. 发布后只读验收

- [ ] `git ls-remote` 显示签名 tag 指向预期 source commit。
- [ ] GitHub hosted CI 链接、commit、tag 和 Release assets 可从干净环境读取。
- [ ] HF repo 页面显示 current model card，下载链接固定到完整 immutable commit。
- [ ] 从 HF immutable commit 下载后的 `checksums.txt`、离线加载和最小前向再次通过。
- [ ] GitHub Release 的 publication receipt 能解析并匹配 GitHub/HF 两端实际 SHA。
- [ ] GitHub 和 HF 均没有 wheelhouse、token、真实 PII、本机绝对用户名路径或 preauthorization 文件。
- [ ] Release notes 只保留 synthetic Open24 限定声明，并明确披露 PII Bench 限制。
- [ ] 建立回滚/撤回说明：发现许可证、安全、身份或 checksum 问题时先停止分发、保留证据，再发布
  更正或撤回通知；不得无记录覆盖 tag 或 immutable commit。

## 6. 本轮明确未执行

- 未创建 `whyiug/pii-detect-model`；
- 未添加 Git remote，未 push，未创建或推送 tag，未创建 GitHub Release；
- 仅只读验证了现有 Hugging Face 登录和两个候选 repo id；未修改认证配置、未创建 HF repo、未上传模型；
- 未上传 wheel、wheelhouse、SBOM 或任何发布回执；
- 未修改冻结的 `model-package-v2r2`，也未把它包装成已发布工件。

下一次真正执行远端写入前，应先把本清单的未完成项、目标 namespace、repo 可见性、最终 source
commit、签名 tag 与 publication-successor receipt 交由维护者复核并明确授权。
