# syntax=docker/dockerfile:1
#
# Parameterized PyTorch image for CI and ad-hoc testing.
# The devcontainer does not use this file; it is standalone.
#
# Build:
#   docker build -f docker/Dockerfile --build-arg PYTORCH_TAG=latest -t flowc .
#
# Tested tags (pytorch/pytorch):
#   2.13.0-cuda12.6-cudnn9-runtime
#   latest

ARG PYTORCH_TAG=latest
FROM pytorch/pytorch:${PYTORCH_TAG}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /flowc

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY pyproject.toml README.md LICENSE.md ./
COPY flowcon ./flowcon

# CI images run as root, so installing into the image Python is fine.
# torch is already provided by the base image and will not be replaced
# unless --upgrade is passed.
RUN uv pip install --system --no-cache -e ".[dev]"

COPY tests ./tests

CMD ["/bin/bash"]
