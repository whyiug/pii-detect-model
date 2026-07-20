# Docker deployment

Both images run as an unprivileged user and default to Hugging Face/Transformers offline mode. They
contain project code only: model weights, credentials and deployment-specific configuration must be
provided at runtime. The inference image declares no public port, and its build context uses a
runtime-source allowlist which excludes local caches and generated package metadata.

Build from the repository root:

```bash
docker build -f docker/train.Dockerfile -t pii-zh-qwen-train:local .
docker build -f docker/inference.Dockerfile -t pii-zh-qwen-inference:local .
```

An uncached build needs network access for the base image and locked Python dependencies. It does
not download a Hugging Face model or require a Hugging Face token.

The inference build defaults to `RUNTIME_PROFILE=rules-only`. The profiles intentionally have
different dependency surfaces:

| Profile | Installed project extras | Intended path | Deliberately absent |
|---|---|---|---|
| `rules-only` | `service` | Direct rules CLI/Python and default HTTP factory | Presidio, torch, Transformers, CUDA libraries and model weights |
| `local-model` | `cascade,service` | Explicit, separately mounted local model plus cascade/HTTP | Model weights and credentials |

Presidio remains an optional integration, but is not pulled into the lightweight default image
because its analyzer dependency brings a larger NLP stack that the direct rules pipeline does not
execute. To create an image which can load a separately mounted local model, select the
`local-model` profile and use a separate tag:

```bash
docker build -f docker/inference.Dockerfile \
  --build-arg RUNTIME_PROFILE=local-model \
  -t pii-zh-qwen-inference:model-local .
```

Only `rules-only` and `local-model` are accepted profile values. The build argument controls
dependencies, not a host filesystem path: do not send model weights through the build context.
Mount the model directory read-only when the container starts.

For training, mount the prepared manifests and local base checkpoint read-only, and keep outputs in
a separate writable directory:

```bash
docker run --rm --gpus 'device=<free-gpu-id>' \
  -v /approved/checkpoint:/inputs/base:ro \
  -v /approved/manifests:/inputs/manifests:ro \
  -v "$PWD/runs/container-output:/outputs" \
  pii-zh-qwen-train:local <explicit train.py arguments>
```

The Dockerfile does not choose or reserve a GPU. On a shared server, inspect GPU usage first and
select a device without interrupting other workloads. The default inference image installs only the
base and HTTP service runtime dependencies; its default command runs a CPU-only, offline rules
example:

```bash
docker run --rm pii-zh-qwen-inference:local
```

For a local model, mount the downloaded Hugging Face directory read-only and explicitly select the
model profile, mode and device:

```bash
MODEL_DIR=/absolute/path/to/pii-zh-qwen3-0.6b-24class
docker run --rm \
  -v "$MODEL_DIR:/models/model:ro" \
  pii-zh-qwen-inference:model-local \
  pii-zh detect \
    --profile community-model-cascade-v1 \
    --mode cascade \
    --model-path /models/model \
    --device cpu \
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
