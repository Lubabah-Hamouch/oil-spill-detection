# Python API serving the models. The frontend is built and deployed separately (see README, Deployment).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

# Run as an unprivileged user, with a home directory for any library caches.
RUN useradd --create-home --uid 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /app

# CPU-only torch/torchvision first (versions taken from requirements.txt), then the rest.
COPY requirements.txt .
RUN pip install $(grep -E '^(torch|torchvision)==' requirements.txt) \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
RUN chown -R user:user /app

USER user

# Bake the trained weights into the image. download_weights.py does not fail on errors, so check the files exist.
RUN python scripts/download_weights.py \
    && test -s models/classification_best.pth \
    && test -s models/segmentation_best.pth

# Memory: the app runs inference on one CPU thread (see src/api/main.py), so keep OpenMP from
# starting a thread pool, and stop glibc giving each request thread its own malloc arena.
ENV OMP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
