# syntax=docker/dockerfile:1
# Self-contained serving image: the build stage trains on synthetic data, runs the promotion gate
# and exports the @champion; the runtime stage only carries the venv and the exported bundle.

FROM python:3.11-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 MLFLOW_DISABLE_AGENT_HINT=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt pyproject.toml README.md ./
RUN pip install -r requirements.txt
COPY src ./src
RUN pip install --no-deps . \
 && fraud-mlops generate \
 && fraud-mlops train \
 && fraud-mlops gate --strict \
 && fraud-mlops export --out /build/model

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1 \
    PATH="/opt/venv/bin:$PATH" MODEL_PATH=/app/model HOST=0.0.0.0 PORT=8000
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system app && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app
COPY --from=build /opt/venv /opt/venv
COPY --from=build --chown=app:app /build/model /app/model
WORKDIR /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["fraud-mlops", "serve"]
