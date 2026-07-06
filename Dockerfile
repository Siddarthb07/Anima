FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY core ./core
COPY alignment ./alignment
COPY probes ./probes
COPY api ./api
COPY cli ./cli
COPY benchmarks ./benchmarks
COPY scripts ./scripts

# Prefer CPU-only torch (~200 MB). Fall back to PyPI with long timeouts if the PyTorch CDN is unreachable.
RUN pip install --no-cache-dir pip setuptools wheel \
    && (pip install --no-cache-dir --default-timeout=120 \
        --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org \
        torch --index-url https://download.pytorch.org/whl/cpu \
        || pip install --no-cache-dir --default-timeout=3600 --retries 10 "torch>=2.1.0") \
    && pip install --no-cache-dir --default-timeout=3600 --retries 10 .

ENV PYTHONUNBUFFERED=1
ENV ANIMA_API_PORT=8010
EXPOSE 8010

CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8010"]
