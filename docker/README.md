# Offline Docker templates

Both images install project code only. They set Hugging Face/Transformers offline mode, contain no
model weight or credential, declare no network port, and run as an unprivileged user. Python and
uv default to pinned versions, and `uv sync --frozen` installs from the committed lock. The
inference image copies `uv` from a digest-pinned official Astral image instead of downloading a uv
wheel with `pip`; `UV_PYTHON_DOWNLOADS=never` also prevents uv from silently fetching a different
Python runtime. For a production rebuild, additionally override `PYTHON_IMAGE` with the reviewed
immutable image digest.

The inference build context is a strict allowlist. It uses the minimal
[`PACKAGE_README.md`](PACKAGE_README.md) for package metadata instead of copying the repository root
README, so host-specific model paths and credential setup instructions do not enter an image layer.
The allowlist re-excludes `src/**/__pycache__/` and `src/**/*.py[cod]`; local bytecode can embed the
absolute path of the machine which compiled it and must never be copied into a portable image. It
also excludes locally generated `*.egg-info`/`*.dist-info` metadata because those files can retain
an older root README even after the container package README has been minimized.

Build from the repository root:

```bash
docker build -f docker/train.Dockerfile -t pii-zh-qwen-train:local .
docker build -f docker/inference.Dockerfile -t pii-zh-qwen-inference:local .
```

The build still needs network access for an uncached Python base image and the exact packages in
`uv.lock`; it never needs a Hugging Face token or model download. On a shared host, bound a smoke
build and use a task-specific tag rather than leaving an unlimited downloader behind:

```bash
timeout --signal=INT --kill-after=15s 600s \
  docker build --progress=plain -f docker/inference.Dockerfile \
  -t piizh-smoke-rules:local .
```

### Bounded local verification on 2026-07-15

The final portable form of the lightweight default was built and exercised on the shared
development host with only synthetic input:

- transient tag `piizh-smoke-rules:20260715-portable2`, image ID
  `sha256:c455d42c83dbc824fc5eb34d8569950ff75f13b1edfce7ca59e8b82f632c853c`;
- the first cold locked-dependency build completed in `519.96s` under the 600-second bound; almost
  all elapsed time was the public package source downloading the locked 2 MiB `pydantic-core`
  wheel. After context hardening, the dependency-cache-backed final rebuild took `3.41s`;
- the cache-backed final rebuild reported an incremental BuildKit context transfer of `8.69kB`;
  the full inference allowlist is `2,861,521` raw regular-file bytes before tar/compression. Both
  the intended context files and serialized final image layers had zero task-owner home or shared
  model-root path matches;
- final image size `207,282,151` bytes (`207MB` in `docker image ls`), runtime user `app`, no
  declared `EXPOSE`, and offline Hub/Transformers environment enabled;
- the default CLI found the synthetic `demo@example.com` span and returned only offsets, type,
  score, source and decision steps;
- `torch`, `transformers`, `presidio_analyzer` and `spacy` were all absent by runtime import-spec
  check;
- a short-lived container published `8000/tcp` only as host `127.0.0.1:32780`; `/healthz` returned
  `status=ok, mode=rules-only`, and `/v1/analyze` returned the expected email offset without the
  matched value;
- the task container and transient image were removed after the smoke. No task-specific container,
  image, network or build process remained.

This is a local engineering smoke, not a retained release image, signed artifact, performance
benchmark, Presidio-container claim or GPU/model parity result.

The inference build defaults to `RUNTIME_PROFILE=rules-only`. The profiles intentionally have
different dependency surfaces:

| Profile | Installed project extras | Intended path | Deliberately absent |
|---|---|---|---|
| `rules-only` | `service` | Direct rules CLI/Python and default HTTP factory | Presidio, torch, Transformers, CUDA libraries and model weights |
| `local-model` | `cascade,service` | Explicit, separately mounted local model plus cascade/HTTP | Model weights and credentials |

Presidio remains an optional wheel integration (`pii-zh-qwen[presidio]`), but is not pulled into the
lightweight default image because its analyzer dependency also brings a large NLP stack that the
direct rules pipeline does not execute. To create an image which can load a separately mounted local
model, opt in explicitly and use a separate tag:

```bash
docker build -f docker/inference.Dockerfile \
  --build-arg RUNTIME_PROFILE=local-model \
  -t pii-zh-qwen-inference:model-local .
```

Only `rules-only` and `local-model` are accepted profile values. The build argument controls
dependencies, not a host filesystem path: do not send a model path or weight through the build
context. Mount the approved directory read-only when the container starts. If a deployment needs
Presidio without a local model, build a separately reviewed image/wheel environment rather than
silently expanding the default rules-only profile.

Run training only with approved prepared manifests and a local base checkpoint mounted read-only;
use an output directory owned by the current task:

```bash
docker run --rm --gpus 'device=<free-gpu-id>' \
  -v /approved/checkpoint:/inputs/base:ro \
  -v /approved/manifests:/inputs/manifests:ro \
  -v "$PWD/runs/container-output:/outputs" \
  pii-zh-qwen-train:local <explicit train.py arguments>
```

Inspect `nvidia-smi` first, select an idle GPU, and never stop or kill another GPU process. The
Dockerfile does not choose or reserve a GPU. The default inference image installs only the base and
HTTP service runtime dependencies; its default command runs a CPU-only, offline synthetic rules
smoke test:

```bash
docker run --rm pii-zh-qwen-inference:local
```

For a local model, mount an already built HF release read-only and explicitly select a mode/device:

```bash
docker run --rm \
  -v /approved/hf-release:/models/model:ro \
  pii-zh-qwen-inference:model-local \
  pii-zh detect --mode cascade --model-path /models/model --device cpu \
  --text '测试邮箱 demo@example.com'
```

The template intentionally declares no `EXPOSE` and starts no service by default. For a short-lived
local HTTP smoke test, publish only host loopback; binding `0.0.0.0` below is limited to the container
namespace so Docker can forward it to `127.0.0.1` on the host:

```bash
docker run --rm -p 127.0.0.1:18080:8000 \
  pii-zh-qwen-inference:local \
  uvicorn --factory pii_zh.service.app:create_app \
  --host 0.0.0.0 --port 8000 --workers 1 --no-access-log
```

Stop that task with `Ctrl-C`; `--rm` removes only this task's container. Do not publish the port on
all host interfaces. A model-enabled service additionally needs the read-only model mount and an
explicit application module which constructs `CascadePipeline.from_pretrained()`.
