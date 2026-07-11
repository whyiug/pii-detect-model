# syntax=docker/dockerfile:1.7
ARG PYTHON_IMAGE=python:3.12.12-slim-bookworm
FROM ${PYTHON_IMAGE}

ARG UV_VERSION=0.10.4

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH=/workspace/.venv/bin:$PATH

RUN groupadd --system app && useradd --system --gid app --create-home app
WORKDIR /workspace

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}" && \
    uv sync --frozen --no-dev --extra training --extra presidio --no-editable

RUN mkdir -p /models && chown app:app /models
USER app

# No EXPOSE and no server entrypoint: mount /models/model read-only and supply a local command.
CMD ["python", "-c", "from pathlib import Path; assert Path('/models/model/model.safetensors').is_file(), 'mount a local safetensors release at /models/model' "]
