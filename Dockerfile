# Official uv image: Debian slim + Python 3.12 + uv preinstalled.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Production builds use ./global_model. CI overrides this with its deterministic
# full-universe fixture; neither image contains an AAPL-only runtime.
ARG MODEL_DIR=global_model
ARG MODEL_VARIANT=production
LABEL org.opencontainers.image.model-variant="${MODEL_VARIANT}"

# --- Dependency layer (cached) ---
# Copy ONLY the lockfiles first. This layer is rebuilt only when deps change,
# so editing app code later doesn't trigger a full reinstall (Docker layer caching).
COPY pyproject.toml uv.lock ./
# --frozen: fail if the lock is stale (reproducible). --no-dev: skip pytest/ruff/httpx.
# --no-install-project: install deps only, not our package — we put src on PYTHONPATH instead.
RUN uv sync --frozen --no-dev --no-install-project

# --- App layer ---
# Copy the code, config, and the baked-in global model release.
COPY src ./src
COPY configs ./configs
COPY ${MODEL_DIR} ./model

# Put the venv's binaries on PATH and make `src` importable without installing the package.
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
# Load the global model from the baked-in folder — no mlflow.db in the container.
ENV MODEL_URI="/app/model"
ENV MODEL_DIR="/app/model"

# Document the port the app listens on.
EXPOSE 8000

# --host 0.0.0.0 is REQUIRED in a container: bind all interfaces, not just localhost,
# or the port mapping from the host won't reach the app.
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
