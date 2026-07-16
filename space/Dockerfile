# Anima HF Space — original React dashboard + FastAPI (same container).
# Builds dashboard from GitHub main; serves on port 7860.

FROM node:22-bookworm AS dashboard
WORKDIR /src
RUN git clone --depth 1 https://github.com/Siddarthb07/Anima.git repo
WORKDIR /src/repo/dashboard
ENV VITE_SAME_ORIGIN=1
ENV VITE_DEFAULT_MODEL=Qwen/Qwen2.5-0.5B-Instruct
ENV VITE_API_HTTP_TARGET=
ENV VITE_WS_BASE=
RUN npm ci && npm run build

FROM python:3.11-slim-bookworm
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# CPU torch first (avoid huge CUDA wheels on free Space hardware).
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --default-timeout=600 \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.5.1 \
 && pip install --no-cache-dir "git+https://github.com/Siddarthb07/Anima.git"

COPY --from=dashboard /src/repo/dashboard/dist /app/dashboard/dist
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONUNBUFFERED=1
ENV ANIMA_PUBLIC_MODE=1
ENV ANIMA_FORCE_CPU=1
ENV ANIMA_SERVE_DASHBOARD=/app/dashboard/dist
ENV ANIMA_WARMUP_MODEL=Qwen/Qwen2.5-0.5B-Instruct
ENV ANIMA_MAX_NEW_TOKENS=64

EXPOSE 7860
CMD ["/app/entrypoint.sh"]
