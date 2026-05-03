FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY core ./core
COPY alignment ./alignment
COPY probes ./probes
COPY api ./api
COPY cli ./cli

RUN pip install --no-cache-dir pip setuptools wheel \
    && pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1

ENV ANIMA_API_PORT=8010
EXPOSE 8010

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8010"]
