# fastroute routing API.
#
#   docker build -t fastroute .                 # fetches central London from OpenStreetMap
#   docker run -p 8000:8000 fastroute           # API on http://localhost:8000, docs at /docs
#
# Build with --build-arg FETCH=0 to skip the download and mount your own
# network at /app/data/central-london.npz instead.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTROUTE_NETWORK=/app/data/central-london.npz

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY fastroute ./fastroute
RUN pip install ".[api,ml]" && useradd --create-home --uid 1000 fastroute \
    && mkdir -p data && chown fastroute data
USER fastroute

ARG FETCH=1
RUN if [ "$FETCH" = "1" ]; then \
      python -m fastroute fetch && python -m fastroute build && rm -rf data/raw; \
    fi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["sh", "-c", "uvicorn fastroute.api:app_from_env --factory --host 0.0.0.0 --port 8000 --workers ${WORKERS:-2}"]
