FROM python:3.13-slim


# Install uv by copying it directly from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app


# Copy dependency files first for Docker layer caching
COPY pyproject.toml uv.lock ./

# Install only runtime dependencies  , not install project package
RUN uv sync --locked --no-dev --no-install-project


# Copy the project into the image with the model -> not good practice
COPY . .


# Install the project itself if your pyproject.toml defines it as a package
RUN uv sync --locked --no-dev


# Copy the trained model
COPY models/model.pkl ./models/model.pkl

ENV PATH="/app/.venv/bin:$PATH"
ENV PRODML_MODEL_PATH="/app/models/model.pkl"



HEALTHCHECK \
    --interval=30s \
    --timeout=3s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"



EXPOSE 8000


# Run the application.
CMD ["uvicorn", "prodml.api.main:app", "--host","0.0.0.0","--port","8000"]
