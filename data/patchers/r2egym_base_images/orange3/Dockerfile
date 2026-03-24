FROM python:3.10-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    git curl wget jq \
    build-essential \
    libffi-dev libssl-dev \
    locales locales-all tzdata \
    tmux \
    libxml2-dev libxslt1-dev libgl1-mesa-glx libglib2.0-0 \
    xvfb libxkbcommon-x11-0 libdbus-1-3 libegl1 libxcb-xinerama0 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
    libxcb-render-util0 libxcb-shape0 libxcb-cursor0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip uv

# Pre-install PyQt5 and orange3 deps (the main pain point)
RUN pip install PyQt5 PyQtWebEngine 2>/dev/null || true
RUN pip install orange3 2>/dev/null || true

# Clone the repo
RUN git clone https://github.com/biolab/orange3.git /testbed
WORKDIR /testbed

# Install in editable mode with test extras
RUN pip install -e '.[test]' 2>/dev/null || \
    pip install -e . 2>/dev/null || true

RUN pip install pytest pytest-xdist 2>/dev/null || true

RUN mkdir -p /logs /r2e_tests
