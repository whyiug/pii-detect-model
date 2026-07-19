# syntax=docker/dockerfile:1.7
ARG PYTHON_IMAGE=python:3.12.12-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.28-python3.12-trixie-slim@sha256:3137a0b606f65a74ee0245f43dae219b09e8af98fc37fef20841cbceef35a646

FROM ${UV_IMAGE} AS uv-bin
FROM ${PYTHON_IMAGE}

ARG RUNTIME_PROFILE=rules-only

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/workspace/.venv/bin:$PATH

RUN groupadd --system app && useradd --system --gid app --create-home app
WORKDIR /workspace

COPY --from=uv-bin /usr/local/bin/uv /usr/local/bin/uv
COPY --chmod=0644 pyproject.toml uv.lock LICENSE ./
COPY --chmod=0644 docker/PACKAGE_README.md ./README.md
COPY src ./src
RUN --mount=type=cache,target=/var/cache/uv \
    case "${RUNTIME_PROFILE}" in \
      rules-only) \
        uv sync --frozen --no-dev --extra service --no-editable \
        ;; \
      local-model) \
        uv sync --frozen --no-dev --extra cascade --extra service --no-editable \
        ;; \
      *) \
        echo "unsupported RUNTIME_PROFILE: ${RUNTIME_PROFILE}" >&2; exit 2 \
        ;; \
    esac

RUN mkdir -p /models && chown app:app /models
USER app

# No EXPOSE and no server entrypoint. The default command is an offline synthetic smoke test.
CMD ["pii-zh", "detect", "--text", "测试邮箱 demo@example.com"]
