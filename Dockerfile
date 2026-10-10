# syntax=docker/dockerfile:1
# Two targets on CPU PyTorch:
#   runtime  diagnoses a run report:
#     docker build --target runtime -t runsleuth .
#     docker run --rm -v "$PWD/examples:/work/examples:ro" runsleuth diagnose \
#       --run examples/frozen_head/run_report.json \
#       --reference examples/clean/run_report.json --no-llm
#   test     runs the whole test suite:
#     docker build --target test -t runsleuth-test . && docker run --rm runsleuth-test

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# CPU wheels of the pinned PyTorch: diagnosis and the tests need no GPU.
RUN pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu

FROM base AS runtime
WORKDIR /opt/runsleuth
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[llm]"
RUN useradd --create-home runsleuth && mkdir /work && chown runsleuth /work
USER runsleuth
WORKDIR /work
ENTRYPOINT ["runsleuth"]
CMD ["--help"]

FROM base AS test
# Git is there for the held-out tool, which records the commit a seed draw froze.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/runsleuth
COPY . .
RUN pip install -e ".[dev,llm,hf,camelyon]"
CMD ["python", "-m", "pytest"]
