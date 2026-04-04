FROM python:3.11-slim

# HF Spaces requires a non-root user with UID 1000
RUN useradd -m -u 1000 user

WORKDIR /app

# Install dependencies first (layer caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application code
COPY --chown=user . /app

# Switch to non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# HF Spaces default port
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]