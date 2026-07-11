# Offline Docker templates

Both images install project code only. They set Hugging Face/Transformers offline mode, contain no
model weight or credential, declare no network port, and run as an unprivileged user. Python and
uv default to patch-pinned versions, and `uv sync --frozen` installs from the committed lock; for a
production rebuild, additionally override `PYTHON_IMAGE` with the reviewed immutable image digest.

Build from the repository root:

```bash
docker build -f docker/train.Dockerfile -t pii-zh-qwen-train:local .
docker build -f docker/inference.Dockerfile -t pii-zh-qwen-inference:local .
```

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
Dockerfile does not choose or reserve a GPU. Inference likewise requires an already built local HF
directory:

```bash
docker run --rm -v /approved/hf-release:/models/model:ro \
  pii-zh-qwen-inference:local
```

Supply an application command only when it binds to `127.0.0.1`; the template intentionally does
not expose or start a service.
