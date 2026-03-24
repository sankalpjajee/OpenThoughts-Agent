FROM python:3.11-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    git curl wget jq \
    build-essential \
    libffi-dev libssl-dev \
    locales locales-all tzdata \
    tmux \
    libjpeg-dev zlib1g-dev libpng-dev libtiff-dev libwebp-dev \
    libfreetype6-dev liblcms2-dev libopenjp2-7-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip uv

# Pre-install Pillow wheel (fast)
RUN pip install Pillow

# Clone the repo
RUN git clone https://github.com/python-pillow/Pillow.git /testbed
WORKDIR /testbed

# Install in editable mode with test extras
RUN pip install -e '.[tests]' 2>/dev/null || \
    pip install -e . 2>/dev/null || true

RUN pip install pytest pytest-xdist 2>/dev/null || true

RUN mkdir -p /logs /r2e_tests
