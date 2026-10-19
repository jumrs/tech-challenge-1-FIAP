# syntax=docker/dockerfile:1

FROM python:3.13-slim AS builder

# uv fixado na mesma versao do desenvolvimento, copiado da imagem oficial (sem pip install).
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /bin/uv

# Usa o Python da imagem (nao baixa outro), compila .pyc e copia os arquivos para o .venv.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Dependencias primeiro, para aproveitar o cache de camadas quando so o codigo muda.
# --no-default-groups: sem os grupos dev (pytest) e training (MLflow, KaggleHub, graficos).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-install-project

# Codigo e instalacao do pacote churn (nao editavel: src/ nao e necessario no runtime).
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-editable


FROM python:3.13-slim

LABEL org.opencontainers.image.title="churn-api" \
      org.opencontainers.image.description="API de predicao de churn (Cell2Cell) - Tech Challenge Fase 1 FIAP" \
      org.opencontainers.image.version="1.0.0"

# libgomp1: runtime OpenMP exigido pelo LightGBM (nao vem na imagem slim).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Usuario sem privilegios para rodar a API.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# PATH com o .venv; CHURN_PROJECT_ROOT faz o config.py achar /app/models/champion.joblib.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHURN_PROJECT_ROOT=/app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
# O modelo precisa existir antes do build: `uv run python -m churn.main`.
COPY --chown=appuser:appuser models/champion.joblib /app/models/champion.joblib

USER appuser

EXPOSE 8000

# Healthcheck sem curl: usa o proprio Python da imagem.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"]

CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
