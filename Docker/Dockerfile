FROM python:3.13-slim  AS Builder


# Install uv by copying it directly from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /build

# Use the Python provided by our base image.
ENV UV_PYTHON_DOWNLOADS=0

# Avoid hardlink warnings when using a cache mount.
ENV UV_LINK_MODE=copy


# Copy dependency files first for Docker layer caching
COPY ../pyproject.toml ../uv.lock ./

# Install only runtime dependencies  , not install project package
RUN  --mount=type=cache,target=/root/.cache/uv \
         uv sync --locked --no-dev --no-install-project


# Copy the project into the image
COPY ../src/ ./src/



# Install our application as a regular package.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev \
        --no-editable






FROM python:3.13-slim AS runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Prevent Python from writing .pyc files at runtime.
ENV PYTHONDONTWRITEBYTECODE=1

# Use executables from the uv-created environment.
ENV PATH="/app/.venv/bin:$PATH"

# Create a non-root application user.
RUN groupadd --system appgroup \
    && useradd \
        --system \
        --gid appgroup \
        --home-dir /app \
        appuser


# Copy only the installed environment from the builder.
COPY --from=builder /app/.venv /app/.venv

# Include our trusted, locally trained model.
COPY --chown=appuser:appgroup models/model.pkl /app/models/model.pkl


# Explicit model path for our Pydantic settings.
ENV PRODML_MODEL_PATH=/app/models/model.pkl
ENV PRODML_LOG_LEVEL=INFO



# # Run the service without root privileges.
USER appuser

EXPOSE 8000



HEALTHCHECK \
    --interval=30s \
    --timeout=3s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"


# Run the application.
CMD ["uvicorn", "prodml.api.main:app", "--host","0.0.0.0","--port","8000"]
