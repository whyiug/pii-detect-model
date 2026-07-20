# GitHub / Hugging Face 发布手册（community v2 RC1）

本手册只适用于 `pii-zh-qwen 0.2.0rc1`。发布按“源码冻结 → 本地制品 → private HF 回读 →
GitHub draft → pre-publication gate → 单独公开确认”执行。任何一步身份、哈希、可见性或测试不一致
都立即停止；不得移动已推送 tag、覆盖 evidence，或把 private HF 自动切成 public。

## 1. 当前状态与硬停止条件

截至 2026-07-20：

- [x] GitHub 源码仓库 `whyiug/pii-detect-model` 已为 public；默认发布分支为
  `agent/community-v2-release-prep`。
- [x] GitHub Private Vulnerability Reporting 已启用。
- [x] private HF model repo `Forrest20231206/pii-zh-qwen3-0.6b-24class` 已创建；当前尚未上传模型。
- [x] 维护者 `whyiug` 已审批 `LICENSE`、`NOTICE`、`THIRD_PARTY_NOTICES.md` 与许可证清单，无例外。
- [x] 许可证回执语义 SHA-256 为
  `0d1862151175c36e0ec262f381ce2e1af0c002e7131ed80f035881397b4cc738`，文件 SHA-256 为
  `501fd5080a9748030d9875d331b24d479eca90fff483e911ec2410b35489619c`。
- [x] 新 HF token 已从仓库外、权限受限的 env 文件做过只读身份/写权限验证；本手册不记录该文件的
  本机路径，也不记录 token、前缀、长度或摘要。上传时仍须重新验证。
- [x] commit `a6284fe6ebe6854133feb0c5a5932ee2062fa578` 的 hosted CI 已通过；它只是历史
  preflight，不能替代最终 source commit 的 CI。
- [ ] 独立、无管理权限 GitHub 账号尚未提交纯合成 private vulnerability report；维护者也尚未
  记录 accepted test receipt。当前 API 清单为空。
- [ ] 本机 SSH 公钥可在本地生成有效 Git tag 签名，但尚未登记为 GitHub SSH signing key；必须先
  完成交互 OAuth scope `write:ssh_signing_key` 并核验 fingerprint
  `SHA256:j3DjYC57yPDuXhn5qbIKYmc0OLyJrRtbqgvsLPIl4hg`。
- [ ] 最终 source commit、与其精确匹配的 hosted CI、签名 tag、clean wheel、SBOM、发布 checksums、
  publication-successor、private HF immutable revision 和 GitHub draft Release 尚未创建。

以上未完成项都是硬停止条件。尤其禁止上传旧的 `$RELEASE_RUN/model-package-v2r2`：它包含旧
`SECURITY.md`、旧模型卡和 preauthorization 文件，只能作为只读源证据。

## 2. 固定变量

```bash
# 两个值只在执行机本地设置；不要把绝对路径提交到公开仓库。
: "${RELEASE_RUN:?set RELEASE_RUN to the reviewed external release-run directory}"
: "${HF_TOKEN_ENV_FILE:?set HF_TOKEN_ENV_FILE to the external private env file}"
export RELEASE_RUN HF_TOKEN_ENV_FILE
export GH_REPO="whyiug/pii-detect-model"
export RELEASE_BRANCH="agent/community-v2-release-prep"
export RELEASE_TAG="v0.2.0rc1"
export HF_REPO_ID="Forrest20231206/pii-zh-qwen3-0.6b-24class"

export SOURCE_PACKAGE="$RELEASE_RUN/model-package-v2r2"
export PUBLICATION_DIR="$RELEASE_RUN/model-package-publication-successor-v1"
export PUBLICATION_MODEL_CARD="$RELEASE_RUN/README.publication.final.md"
export FINAL_RELEASE_NOTES="$RELEASE_RUN/RELEASE_NOTES.publication.final.md"
export DIST_DIR="$RELEASE_RUN/dist-publication-successor-v1"
export BUILD_TREE="$RELEASE_RUN/source-build-publication-successor-v1"

export FINAL_LOCAL_RECEIPT="$RELEASE_RUN/community-cascade-release-v2.receipt.v2r3.json"
export LICENSE_APPROVAL_RECEIPT="$RELEASE_RUN/human-license-approval-receipt.json"
export SECURITY_CHANNEL_RECEIPT="$RELEASE_RUN/tested-private-security-channel-receipt.json"
export LOCAL_PACKAGE_VERIFICATION="$RELEASE_RUN/publication-package-local-verification.json"
export HF_DOWNLOAD_DIR="$RELEASE_RUN/hf-immutable-download-v1"
export HF_CACHE_DIR="$RELEASE_RUN/hf-cache-publication-v1"
export HF_DOWNLOAD_PROVENANCE="$RELEASE_RUN/hf-download-provenance.v1.json"
export HF_PACKAGE_VERIFICATION="$RELEASE_RUN/hf-package-verification.v1.json"
export GITHUB_EVIDENCE="$RELEASE_RUN/github-prepublication-evidence.v1.json"
export HF_EVIDENCE="$RELEASE_RUN/hf-prepublication-evidence.v1.json"
export PREPUBLICATION_RECEIPT="$RELEASE_RUN/community-v2-pre-publication-gate-receipt.json"

export WHEEL_PATH="$DIST_DIR/pii_zh_qwen-0.2.0rc1-py3-none-any.whl"
export SBOM_PATH="$DIST_DIR/sbom.cdx.json"
export RELEASE_CHECKSUMS="$DIST_DIR/checksums.txt"
export GITHUB_RELEASE_ASSETS_RECEIPT="$RELEASE_RUN/community-v2-github-release-assets-receipt.json"
export POST_ATTACHMENT_VERIFICATION="$RELEASE_RUN/community-v2-post-attachment-verification.json"
export FINAL_POST_ATTACHMENT_RECHECK="$RELEASE_RUN/community-v2-post-attachment-final-recheck.json"
```

所有新输出均为 no-clobber。重跑前不要删除或覆盖旧 evidence；应使用新目录或先人工审计旧文件。
以下命令必须在同一个非交互 Bash 会话中执行，并先启用：

```bash
set -Eeuo pipefail
```

远端状态变更阶段还会额外放入带 `trap` 的 fail-fast 子 shell；任一检查失败都必须停止，不能跳到
后续公开步骤。

## 3. 两项维护者前置操作

### 3.1 独立测试私密报告入口

由无管理权限账号打开：

`https://github.com/whyiug/pii-detect-model/security/advisories/new`

标题：

```text
[CHANNEL TEST] pii-zh-qwen 0.2.0rc1 synthetic reporting test
```

正文：

```text
This is an authorized synthetic end-to-end test of Private Vulnerability
Reporting. It contains no real vulnerability, PII, secret, or exploit.
Please confirm private receipt and close this test report.
```

维护者必须在私密界面实际看到并接受该合成报告。不得放入真实漏洞、PII、secret 或 exploit。随后先
把根 `SECURITY.md` 更新为“已完成独立合成测试”的最终措辞并提交，再用公开 reporter login、接受
时间和 GHSA/test-case ID 生成回执：

```bash
uv run python scripts/build_tested_private_security_channel_receipt.py \
  --output "$SECURITY_CHANNEL_RECEIPT" \
  --security SECURITY.md \
  --package-version 0.2.0rc1 \
  --github-repository "$GH_REPO" \
  --hugging-face-repository "$HF_REPO_ID" \
  --tested-by '<reporter-github-login>' \
  --tested-at '<accepted-rfc3339-time>' \
  --test-case-id '<private-ghsa-or-stable-test-id>' \
  --outcome accepted_private_test_report \
  --attest-private-report-accepted \
  --attest-no-real-sensitive-data \
  --dry-run

uv run python scripts/build_tested_private_security_channel_receipt.py \
  --output "$SECURITY_CHANNEL_RECEIPT" \
  --security SECURITY.md \
  --package-version 0.2.0rc1 \
  --github-repository "$GH_REPO" \
  --hugging-face-repository "$HF_REPO_ID" \
  --tested-by '<reporter-github-login>' \
  --tested-at '<accepted-rfc3339-time>' \
  --test-case-id '<private-ghsa-or-stable-test-id>' \
  --outcome accepted_private_test_report \
  --attest-private-report-accepted \
  --attest-no-real-sensitive-data
```

该回执明确是 `human_attestation_not_remote_verified`，不读取或复制私密报告正文。

### 3.2 登记 SSH signing key

```bash
gh auth refresh -h github.com -s write:ssh_signing_key
```

维护者在浏览器设备页确认后，使用 GitHub REST 将 `~/.ssh/id_ed25519.pub` 登记为 signing key。
不得上传私钥。登记后通过公开 signing-key endpoint 比较 exact public key，并确认 fingerprint 为固定值；
未核验时不得推送 release tag。

## 4. 冻结 source commit、CI 与签名 tag

只精确 stage 本次发布文件，禁止 `git add -A`。工作树中大量未跟踪研究证据属于用户，不进入本次
发布 commit。提交和推送后：

```bash
test "$(git branch --show-current)" = "$RELEASE_BRANCH"
git diff --quiet
git diff --cached --quiet
export SOURCE_COMMIT="$(git rev-parse HEAD)"
test "${#SOURCE_COMMIT}" -eq 40
git push origin "$SOURCE_COMMIT:refs/heads/$RELEASE_BRANCH"
test "$(git ls-remote origin "refs/heads/$RELEASE_BRANCH" | cut -f1)" = "$SOURCE_COMMIT"

gh workflow run ci.yml --repo "github.com/${GH_REPO}" --ref "$RELEASE_BRANCH"
gh run list --repo "github.com/${GH_REPO}" --workflow ci.yml --limit 20 \
  --json databaseId,headSha,status,conclusion,createdAt,event,headBranch,url \
  --jq ".[] | select(
    .event == \"workflow_dispatch\" and
    .headBranch == \"${RELEASE_BRANCH}\" and
    .headSha == \"${SOURCE_COMMIT}\"
  )"
export CI_RUN_ID='<exact-run-id>'
test "$(gh run view "$CI_RUN_ID" --repo "github.com/${GH_REPO}" --json headSha --jq .headSha)" = "$SOURCE_COMMIT"
gh run watch "$CI_RUN_ID" --repo "github.com/${GH_REPO}" --exit-status
test "$(gh run view "$CI_RUN_ID" --repo "github.com/${GH_REPO}" --json conclusion --jq .conclusion)" = success
```

CI 成功后创建 annotated SSH-signed tag；本地验证通过才推送：

```bash
test -z "$(git tag --list "$RELEASE_TAG")"
git tag -s "$RELEASE_TAG" "$SOURCE_COMMIT" -m "pii-zh community v2 RC1"
git tag -v "$RELEASE_TAG"
test "$(git rev-list -n 1 "$RELEASE_TAG")" = "$SOURCE_COMMIT"
git push origin "$RELEASE_TAG"
```

推送后 GitHub tag verification 和本地 SSH cryptographic verification 还会由只读 collector 再核验。
任何源码变更都必须使用新的 RC/tag；不得移动已公开 tag。

## 5. 从 clean source commit 构建 GitHub 制品

```bash
test ! -e "$BUILD_TREE"
test ! -e "$DIST_DIR"
git worktree add --detach "$BUILD_TREE" "$SOURCE_COMMIT"
mkdir "$DIST_DIR"

(
  cd "$BUILD_TREE"
  uv sync --frozen --extra training --extra dev
  uv build --wheel --out-dir "$DIST_DIR"
  uv run python scripts/generate_sbom.py --output "$SBOM_PATH"
  uv run python scripts/generate_sbom.py --output "$SBOM_PATH" --check
)

test -f "$WHEEL_PATH"
test -f "$SBOM_PATH"
test "$(find "$DIST_DIR" -maxdepth 1 -type f -name '*.whl' | wc -l)" -eq 1
(
  cd "$DIST_DIR"
  sha256sum "$(basename "$WHEEL_PATH")" "$(basename "$SBOM_PATH")" > checksums.txt
  sha256sum --check checksums.txt
)
```

随后从 clean worktree 运行 wheel inventory、隔离安装、CLI/service smoke 和 public artifact scan；
不得附带 `wheelhouse/`，也不得把模型权重上传到 GitHub Release。

下面的 producer 会自行检查三个固定文件、重建 wheel inventory 和 SBOM、在禁网 `bwrap`/`prlimit`
沙箱内把这只 exact wheel 安装到临时 venv 后执行 CLI/service smoke，并解包做 public artifact scan；
不接受调用者传入的 PASS：

```bash
(
  cd "$BUILD_TREE"
  uv run --frozen python scripts/build_community_v2_github_release_assets_receipt.py \
    --wheel "$WHEEL_PATH" \
    --sbom "$SBOM_PATH" \
    --checksums "$RELEASE_CHECKSUMS" \
    --source-commit "$SOURCE_COMMIT" \
    --output "$GITHUB_RELEASE_ASSETS_RECEIPT" \
    --dry-run
  uv run --frozen python scripts/build_community_v2_github_release_assets_receipt.py \
    --wheel "$WHEEL_PATH" \
    --sbom "$SBOM_PATH" \
    --checksums "$RELEASE_CHECKSUMS" \
    --source-commit "$SOURCE_COMMIT" \
    --output "$GITHUB_RELEASE_ASSETS_RECEIPT"
)
test "$(stat -c %a "$GITHUB_RELEASE_ASSETS_RECEIPT")" = 444
```

## 6. 构建并验证 publication-successor

先把最终 source/tag/Release URL 渲染进 HF Model Card：

```bash
(
  cd "$BUILD_TREE"
  test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
  git diff --quiet
  git diff --cached --quiet
  uv run --frozen python scripts/render_community_v2_publication_model_card.py \
    --output "$PUBLICATION_MODEL_CARD" \
    --git-source-commit "$SOURCE_COMMIT" \
    --release-tag "$RELEASE_TAG" \
    --github-repository "$GH_REPO" \
    --github-release-url "https://github.com/${GH_REPO}/releases/tag/${RELEASE_TAG}"
)
```

用固定源包和三份外部回执构建新包；权重必须是 reflink/独立 copy，禁止 hardlink：

```bash
COMMON_SUCCESSOR_ARGS=(
  --source-package "$SOURCE_PACKAGE"
  --output "$PUBLICATION_DIR"
  --model-card "$PUBLICATION_MODEL_CARD"
  --security SECURITY.md
  --notice NOTICE
  --third-party-notices THIRD_PARTY_NOTICES.md
  --final-local-receipt "$FINAL_LOCAL_RECEIPT"
  --human-license-approval-receipt "$LICENSE_APPROVAL_RECEIPT"
  --tested-private-security-channel-receipt "$SECURITY_CHANNEL_RECEIPT"
  --git-source-commit "$SOURCE_COMMIT"
  --github-repository "$GH_REPO"
  --hugging-face-repository "$HF_REPO_ID"
)
(
  cd "$BUILD_TREE"
  uv run --frozen python scripts/build_community_v2_publication_successor.py \
    "${COMMON_SUCCESSOR_ARGS[@]}" --dry-run
  uv run --frozen python scripts/build_community_v2_publication_successor.py \
    "${COMMON_SUCCESSOR_ARGS[@]}"
)
```

本地正式验证包含 manifest/checksum 精确闭包、49 个固定 BIO 标签、公开制品扫描，以及在禁网、无
GPU、只读文件系统、单 CPU/受限内存 scope 中的 Transformers 加载与有限前向：

```bash
(
  cd "$BUILD_TREE"
  CUDA_VISIBLE_DEVICES='' uv run --frozen python \
    scripts/verify_community_v2_publication_package.py verify \
    --package-root "$PUBLICATION_DIR" \
    --context local_publication_successor \
    --source-commit "$SOURCE_COMMIT" \
    --github-repository "$GH_REPO" \
    --hugging-face-repository "$HF_REPO_ID" \
    --output "$LOCAL_PACKAGE_VERIFICATION"
)
```

`$PUBLICATION_DIR` 必须精确包含 manifest 声明的文件和受审 `.gitattributes`；不得含 preauthorization、
wheel、wheelhouse、缓存、symlink、额外 binary 或 token。

## 7. 仅上传到 private Hugging Face，并按 immutable SHA 回读

不要 `source` 整个 `.env`。关闭 xtrace，只解析唯一的 `HF_TOKEN`；命令不得打印值：

```bash
set +x
export HF_TOKEN="$(python - <<'PY'
import os
import stat
from pathlib import Path

path = Path(os.environ['HF_TOKEN_ENV_FILE'])
flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0)
fd = os.open(path, flags)
with os.fdopen(fd, encoding='utf-8') as handle:
    metadata = os.fstat(handle.fileno())
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or metadata.st_size <= 0
        or metadata.st_size > 1024 * 1024
    ):
        raise SystemExit('HF token env file must be an owned, private, bounded regular file')
    raw_lines = handle.read().splitlines()
matches = []
for raw in raw_lines:
    line = raw.strip()
    if not line or line.startswith('#'):
        continue
    if line.startswith('export '):
        line = line[7:].lstrip()
    key, separator, value = line.partition('=')
    if separator and key.strip() == 'HF_TOKEN':
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'\"', "'"}:
            value = value[1:-1]
        matches.append(value)
if len(matches) != 1 or not matches[0] or any(ch in matches[0] for ch in '\r\n\0'):
    raise SystemExit('expected exactly one non-empty HF_TOKEN')
print(matches[0], end='')
PY
)"
test -n "$HF_TOKEN"
# 共享服务器同时设置了 HTTP proxy 与 SOCKS ALL_PROXY；项目锁定环境不含
# socksio，因此显式忽略 ALL_PROXY，继续使用现有 HTTP(S) proxy。
unset ALL_PROXY all_proxy
HF_ENDPOINT=https://huggingface.co hf auth whoami
```

上传源只能是 `$PUBLICATION_DIR`：

```bash
HF_ENDPOINT=https://huggingface.co hf upload "$HF_REPO_ID" "$PUBLICATION_DIR" . \
  --repo-type model \
  --commit-message "Stage pii-zh-qwen3-0.6b-24class v0.2.0rc1"

export HF_COMMIT="$(cd "$BUILD_TREE" && HF_ENDPOINT=https://huggingface.co \
  uv run --frozen python - <<'PY'
import os
import re
from huggingface_hub import HfApi

info = HfApi(endpoint='https://huggingface.co', token=os.environ['HF_TOKEN']).repo_info(
    os.environ['HF_REPO_ID'], repo_type='model'
)
if (
    info.id != os.environ['HF_REPO_ID']
    or not info.private
    or not isinstance(info.sha, str)
    or re.fullmatch(r'[0-9a-f]{40}', info.sha) is None
):
    raise SystemExit('private HF immutable identity check failed')
print(info.sha)
PY
)"
```

从官方 endpoint 按完整 SHA 物化新目录；该工具强制比较 remote size、Git blob/LFS OID 与本地字节：

```bash
(
  cd "$BUILD_TREE"
  HF_ENDPOINT=https://huggingface.co uv run --frozen python \
    scripts/materialize_community_v2_hf_snapshot.py \
    --repository "$HF_REPO_ID" \
    --revision "$HF_COMMIT" \
    --output "$HF_DOWNLOAD_DIR" \
    --cache-dir "$HF_CACHE_DIR" \
    --provenance-output "$HF_DOWNLOAD_PROVENANCE"

  CUDA_VISIBLE_DEVICES='' uv run --frozen python \
    scripts/verify_community_v2_publication_package.py verify \
    --package-root "$HF_DOWNLOAD_DIR" \
    --context hugging_face_immutable_download \
    --source-commit "$SOURCE_COMMIT" \
    --github-repository "$GH_REPO" \
    --hugging-face-repository "$HF_REPO_ID" \
    --hugging-face-commit "$HF_COMMIT" \
    --hugging-face-download-provenance "$HF_DOWNLOAD_PROVENANCE" \
    --output "$HF_PACKAGE_VERIFICATION"
)
```

## 8. 创建 GitHub draft Release 与 pre-publication gate

把 HF immutable SHA 渲染进最终 Release body，然后创建 draft；此时 HF 仍为 private：

```bash
(
  cd "$BUILD_TREE"
  uv run --frozen python scripts/render_community_v2_release_notes.py \
    --output "$FINAL_RELEASE_NOTES" \
    --git-source-commit "$SOURCE_COMMIT" \
    --hugging-face-commit "$HF_COMMIT"
)

export REMOTE_TAG_OBJECT="$(
  gh api --hostname github.com --method GET \
    "repos/${GH_REPO}/git/ref/tags/${RELEASE_TAG}" \
    --jq '.object.sha'
)"
test "$(
  gh api --hostname github.com --method GET \
    "repos/${GH_REPO}/git/ref/tags/${RELEASE_TAG}" \
    --jq '.object.type'
)" = tag
test "$(
  gh api --hostname github.com --method GET \
    "repos/${GH_REPO}/git/tags/${REMOTE_TAG_OBJECT}" \
    --jq '.object.sha'
)" = "$SOURCE_COMMIT"

gh release create "$RELEASE_TAG" \
  --repo "github.com/${GH_REPO}" \
  --title "pii-zh community v2 RC1" \
  --notes-file "$FINAL_RELEASE_NOTES" \
  --prerelease \
  --draft \
  "$WHEEL_PATH" "$SBOM_PATH" "$RELEASE_CHECKSUMS"

export GITHUB_RELEASE_ID="$(
  gh api --hostname github.com --method GET \
    "repos/${GH_REPO}/releases/tags/${RELEASE_TAG}" --jq .id
)"
[[ "$GITHUB_RELEASE_ID" =~ ^[1-9][0-9]*$ ]]
test "$(gh api --hostname github.com --method GET \
  "repos/${GH_REPO}/releases/${GITHUB_RELEASE_ID}" --jq .draft)" = true
test "$(gh api --hostname github.com --method GET \
  "repos/${GH_REPO}/releases/${GITHUB_RELEASE_ID}" --jq .prerelease)" = true
test "$(gh api --hostname github.com --method GET \
  "repos/${GH_REPO}/releases/${GITHUB_RELEASE_ID}" --jq .tag_name)" = "$RELEASE_TAG"
```

只读 collector 会复核 public GitHub repo、annotated SSH tag、固定 signing fingerprint、exact hosted
CI、draft body/assets，以及 private HF immutable inventory、下载 provenance 和 package verification：

```bash
export PREPUBLICATION_RECORDED_AT="$(
  python - <<'PY'
from datetime import datetime, timezone

print(datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z'))
PY
)"

(
  cd "$BUILD_TREE"
  uv run --frozen python scripts/collect_community_v2_remote_evidence.py github \
    --source-commit "$SOURCE_COMMIT" \
    --workflow-run-id "$CI_RUN_ID" \
    --release-id "$GITHUB_RELEASE_ID" \
    --output "$GITHUB_EVIDENCE"

  HF_ENDPOINT=https://huggingface.co uv run --frozen python \
    scripts/collect_community_v2_remote_evidence.py hugging-face \
    --download-provenance "$HF_DOWNLOAD_PROVENANCE" \
    --package-verification-receipt "$HF_PACKAGE_VERIFICATION" \
    --output "$HF_EVIDENCE"

  uv run --frozen python scripts/build_community_v2_publication_receipt.py \
    --github-evidence "$GITHUB_EVIDENCE" \
    --github-release-assets-verification-receipt "$GITHUB_RELEASE_ASSETS_RECEIPT" \
    --hugging-face-evidence "$HF_EVIDENCE" \
    --hugging-face-download-verification-receipt "$HF_DOWNLOAD_PROVENANCE" \
    --recorded-at "$PREPUBLICATION_RECORDED_AT" \
    --dry-run

  uv run --frozen python scripts/build_community_v2_publication_receipt.py \
    --github-evidence "$GITHUB_EVIDENCE" \
    --github-release-assets-verification-receipt "$GITHUB_RELEASE_ASSETS_RECEIPT" \
    --hugging-face-evidence "$HF_EVIDENCE" \
    --hugging-face-download-verification-receipt "$HF_DOWNLOAD_PROVENANCE" \
    --recorded-at "$PREPUBLICATION_RECORDED_AT" \
    --output "$PREPUBLICATION_RECEIPT"

  uv run --frozen python scripts/build_community_v2_publication_receipt.py \
    --github-evidence "$GITHUB_EVIDENCE" \
    --github-release-assets-verification-receipt "$GITHUB_RELEASE_ASSETS_RECEIPT" \
    --hugging-face-evidence "$HF_EVIDENCE" \
    --hugging-face-download-verification-receipt "$HF_DOWNLOAD_PROVENANCE" \
    --validate-only "$PREPUBLICATION_RECEIPT"
)

gh release upload "$RELEASE_TAG" --repo "github.com/${GH_REPO}" \
  "$PREPUBLICATION_RECEIPT"

(
  cd "$BUILD_TREE"
  HF_ENDPOINT=https://huggingface.co uv run --frozen python \
    scripts/verify_community_v2_post_attachment.py \
    --github-evidence "$GITHUB_EVIDENCE" \
    --hugging-face-evidence "$HF_EVIDENCE" \
    --prepublication-receipt "$PREPUBLICATION_RECEIPT" \
    --output "$POST_ATTACHMENT_VERIFICATION"
)
test "$(stat -c %a "$POST_ATTACHMENT_VERIFICATION")" = 444
```

回执固定状态为 `READY_FOR_FINAL_PUBLICATION_CONFIRMATION`，不是“已经公开”。post-attachment verifier
再次只读比较四附件的 ID、名称、角色、大小、远端 digest 和下载字节，复核 tag/source/body，且首尾双查
draft=true、HF private=true、main/immutable SHA 与 inventory 未变化；它不下载 HF 权重。

完成本节后清除进程环境中的 token：

```bash
unset HF_TOKEN
```

## 9. 单独最终公开确认

只有维护者在看到以下精确身份后再次明确确认，才执行可见性切换：

- GitHub source commit 与 hosted CI run URL；
- GitHub signing key fingerprint 与 signed tag；
- private HF repo 与 immutable commit；
- wheel/SBOM/checksums 摘要；
- local/HF package verification receipt；
- pre-publication gate receipt SHA-256；
- post-attachment verification receipt SHA-256；
- GitHub draft Release URL。

确认后先给 private HF immutable commit 创建 tag，再公开 HF，核验成功后发布 GitHub prerelease：

```bash
# 重新按第 7 节从仓库外 .env 加载 HF_TOKEN，仍不得打印。
(
  set -Eeuo pipefail
  trap 'unset HF_TOKEN' EXIT
  : "${HF_TOKEN:?HF_TOKEN must be loaded from the external env file}"
  unset ALL_PROXY all_proxy
  cd "$BUILD_TREE"
  test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
  git diff --quiet
  git diff --cached --quiet

  # 最终确认后先重跑只读四附件/HF-private 闭包；它必须发生在可见性切换之前。
  HF_ENDPOINT=https://huggingface.co uv run --frozen python \
    scripts/verify_community_v2_post_attachment.py \
    --github-evidence "$GITHUB_EVIDENCE" \
    --hugging-face-evidence "$HF_EVIDENCE" \
    --prepublication-receipt "$PREPUBLICATION_RECEIPT" \
    --output "$FINAL_POST_ATTACHMENT_RECHECK"
  test "$(stat -c %a "$FINAL_POST_ATTACHMENT_RECHECK")" = 444

  # 先验证 private main，再幂等创建 tag；公开后复核 main/tag/inventory。
  HF_ENDPOINT=https://huggingface.co uv run --frozen python - <<'PY'
import json
import os
import re

from huggingface_hub import HfApi

repo = os.environ['HF_REPO_ID']
commit = os.environ['HF_COMMIT']
tag = os.environ['RELEASE_TAG']
if re.fullmatch(r'[0-9a-f]{40}', commit) is None:
    raise SystemExit('invalid immutable HF commit')
api = HfApi(endpoint='https://huggingface.co', token=os.environ['HF_TOKEN'])
before = api.repo_info(repo, repo_type='model', revision='main')
before_paths = sorted(api.list_repo_files(repo, repo_type='model', revision=commit))
if before.id != repo or before.private is not True or before.sha != commit:
    raise SystemExit('HF private pre-publication state drifted')
api.create_tag(
    repo,
    repo_type='model',
    tag=tag,
    revision=commit,
    tag_message='pii-zh community v2 RC1',
    exist_ok=True,
)
tagged = api.repo_info(repo, repo_type='model', revision=tag)
if tagged.sha != commit:
    raise SystemExit('HF tag does not resolve to immutable commit')
api.update_repo_settings(repo_id=repo, repo_type='model', private=False)
after = api.repo_info(repo, repo_type='model', revision='main')
after_tag = api.repo_info(repo, repo_type='model', revision=tag)
after_paths = sorted(api.list_repo_files(repo, repo_type='model', revision=commit))
if (
    after.id != repo
    or after.private is not False
    or after.sha != commit
    or after_tag.sha != commit
    or after_paths != before_paths
):
    raise SystemExit('HF public post-publication state verification failed')
print(json.dumps({'repo': repo, 'private': False, 'main': commit, 'tag': tag}, sort_keys=True))
PY

  # HF 已复核为 public 后，最后一次确认 GitHub draft 身份与四附件闭包。
  test "$(gh api --hostname github.com --method GET \
    "repos/${GH_REPO}/releases/${GITHUB_RELEASE_ID}" --jq .draft)" = true
  test "$(gh api --hostname github.com --method GET \
    "repos/${GH_REPO}/releases/${GITHUB_RELEASE_ID}" --jq .prerelease)" = true
  test "$(gh api --hostname github.com --method GET \
    "repos/${GH_REPO}/releases/${GITHUB_RELEASE_ID}" --jq .tag_name)" = "$RELEASE_TAG"
  test "$(gh api --hostname github.com --method GET \
    "repos/${GH_REPO}/releases/${GITHUB_RELEASE_ID}/assets?per_page=100" \
    --jq 'map(.name) | sort | join("\n")')" = "$(printf '%s\n' \
      community-v2-pre-publication-gate-receipt.json \
      checksums.txt \
      pii_zh_qwen-0.2.0rc1-py3-none-any.whl \
      sbom.cdx.json | sort)"

  gh api --hostname github.com --method PATCH \
    "repos/${GH_REPO}/releases/${GITHUB_RELEASE_ID}" \
    -F draft=false -F prerelease=true >/dev/null
  test "$(gh api --hostname github.com --method GET \
    "repos/${GH_REPO}/releases/${GITHUB_RELEASE_ID}" --jq .draft)" = false
  test "$(gh api --hostname github.com --method GET \
    "repos/${GH_REPO}/releases/${GITHUB_RELEASE_ID}" --jq .prerelease)" = true
)
```

三步之间逐项只读核验。HF 公开失败时不得发布 GitHub Release；GitHub 发布失败时保留 evidence 并
重试同一 draft，不创建同 tag 的第二个 Release。

## 10. 发布后验收与撤回

- [ ] `git ls-remote` 的 tag 指向 exact source commit，GitHub 显示 verified signature。
- [ ] hosted CI 的 `headSha` 等于 source commit，所有必需 job 成功。
- [ ] HF `main` 和 `v0.2.0rc1` 都解析到记录的 immutable commit；远端 inventory 无额外文件。
- [ ] 从公开 HF immutable SHA 回读后的 checksum、49-label contract、有限前向仍通过。
- [ ] GitHub Release 仅含 wheel、SBOM、checksums 和 pre-publication receipt，无 wheelhouse/权重。
- [ ] 两端无 token、真实 PII、本机用户名绝对路径、preauthorization 或冻结逐行评测数据。
- [ ] Release notes 与 Model Card 只陈述具名合成 Open24 结果，并完整披露公开测试已见和真实数据不足。

若发现许可证、secret、身份、哈希或安全渠道问题，先停止分发并保留 evidence；不得覆盖 immutable
revision 或移动 tag。使用更正/撤回公告说明原因，再发布新的版本身份。
