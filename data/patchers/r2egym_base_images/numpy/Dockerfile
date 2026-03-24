FROM python:3.11-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    git curl wget jq \
    build-essential \
    libffi-dev libssl-dev \
    locales locales-all tzdata \
    tmux \
    gfortran libopenblas-dev liblapack-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip uv

# Pre-install compiled deps from wheels (fast)
RUN pip install numpy cython meson-python meson spin

# Clone the repo
RUN git clone https://github.com/numpy/numpy.git /testbed
WORKDIR /testbed

# Install in editable mode
RUN pip install -e . --no-build-isolation 2>/dev/null || \
    pip install -e . --no-deps 2>/dev/null || true

RUN pip install pytest pytest-xdist hypothesis typing_extensions 2>/dev/null || true

RUN mkdir -p /logs /r2e_tests
