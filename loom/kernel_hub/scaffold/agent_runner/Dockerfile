FROM docker.io/library/node:24-trixie-slim

RUN apt-get update && apt-get install -y python3 python3-pip git curl ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/local/bin/python

RUN npm install -g @openai/codex @anthropic-ai/claude-code

RUN mkdir -p /root/.codex
COPY scaffold/agent_runner/config.toml /root/.codex/config.toml

COPY pyproject.toml /tmp/kernel-hub/pyproject.toml
COPY kernel_evaluator /tmp/kernel-hub/kernel_evaluator
COPY scaffold/__init__.py /tmp/kernel-hub/scaffold/__init__.py
COPY scaffold/client /tmp/kernel-hub/scaffold/client
RUN pip3 install --no-cache-dir --break-system-packages --no-deps /tmp/kernel-hub && rm -rf /tmp/kernel-hub

WORKDIR /workspace
