# ── lbee : image minimale (python:3.12-slim) ────────────────────────────────
# Étape 1 : builder les dépendances dans des wheels isolés.
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# Étape 2 : image finale, utilisateur non-root UID/GID 1000.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LBE_ROOT=/app

# appuser avec UID/GID 1000 : évite tout conflit de droits sur les volumes
# montés depuis l'hôte (./kb, ./class-tools).
RUN groupadd --gid 1000 appuser \
 && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
 && rm -rf /wheels

# Code applicatif.
COPY app ./app
COPY system_prompt.md config.yml ./
COPY class-tools ./class-tools

# Répertoires de travail (./kb est monté au runtime).
RUN mkdir -p /app/kb /app/class-tools \
 && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# /health permet à Docker de vérifier la disponibilité du service.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]