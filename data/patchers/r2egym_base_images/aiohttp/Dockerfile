FROM python:3.11-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    git curl wget jq \
    build-essential \
    libffi-dev libssl-dev \
    locales locales-all tzdata \
    tmux \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip uv

# Pre-install aiohttp wheel and C speedups (fast)
RUN pip install aiohttp[speedups] cython

# Clone the repo
RUN git clone https://github.com/aio-libs/aiohttp.git /testbed
WORKDIR /testbed

# Install in editable mode with dev extras
RUN pip install -e '.[dev]' 2>/dev/null || \
    pip install -e . 2>/dev/null || true

RUN pip install pytest pytest-xdist pytest-aiohttp 2>/dev/null || true

RUN mkdir -p /logs /r2e_tests
