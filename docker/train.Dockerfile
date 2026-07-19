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

COPY pyproject.toml uv.lock LICENSE ./
COPY docker/PACKAGE_README.md ./README.md
COPY src ./src
COPY scripts/train.py ./scripts/train.py
RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}" && \
    uv sync --frozen --no-dev --extra training --no-editable

RUN mkdir -p /inputs /outputs && chown app:app /inputs /outputs
USER app

# Mount the approved base checkpoint and prepared manifests read-only under /inputs.
ENTRYPOINT ["python", "scripts/train.py"]
CMD ["--help"]
