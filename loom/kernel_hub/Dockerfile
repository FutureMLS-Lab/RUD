FROM nvcr.io/nvidia/pytorch:25.06-py3

ARG THUNDERKITTENS_REPO=https://github.com/HazyResearch/ThunderKittens
ARG KERNEL_EVALUATOR_UID=1000
ARG KERNEL_EVALUATOR_GID=1000

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ninja-build curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /home/kernel_evaluator /opt/venv /tmp/kernel_evaluator /tmp/torch-cache/triton /tmp/torch-cache/inductor \
    && chown -R "${KERNEL_EVALUATOR_UID}:${KERNEL_EVALUATOR_GID}" \
        /home/kernel_evaluator /opt/venv /tmp/kernel_evaluator /tmp/torch-cache

RUN git clone --depth 1 "${THUNDERKITTENS_REPO}" /opt/ThunderKittens \
    && chown -R "${KERNEL_EVALUATOR_UID}:${KERNEL_EVALUATOR_GID}" /opt/ThunderKittens

WORKDIR /app

COPY --chown=1000:1000 pyproject.toml ./
COPY --chown=1000:1000 kernel_evaluator ./kernel_evaluator
COPY --chown=1000:1000 replacements ./replacements

RUN chown "${KERNEL_EVALUATOR_UID}:${KERNEL_EVALUATOR_GID}" /app
USER ${KERNEL_EVALUATOR_UID}:${KERNEL_EVALUATOR_GID}
ENV HOME=/home/kernel_evaluator

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/kernel_evaluator/.local/bin:${PATH}"

ENV VIRTUAL_ENV=/opt/venv
RUN uv venv --python 3.14 "${VIRTUAL_ENV}" \
    && uv pip install --no-cache \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -e . "torch==2.11.0+cu128" "nvidia-cutlass-dsl==4.4.2" "quack-kernels>=0.2.10" pytest httpx \
    && ${VIRTUAL_ENV}/bin/python -c "import sys, torch, cutlass; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('cutlass', cutlass.__version__)"

ENV PATH="/opt/venv/bin:${PATH}"

RUN mkdir -p /tmp/torch-cache/triton /tmp/torch-cache/inductor

EXPOSE 8000

CMD ["/opt/venv/bin/uvicorn", "kernel_evaluator.app:app", "--host", "0.0.0.0", "--port", "8000"]
