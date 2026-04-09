FROM python:3.12-slim

# Install Java for OpenDataLoader PDF extraction (requires 11+)
# Debian Trixie (python:3.12-slim base) ships OpenJDK 21 only
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

WORKDIR /app

# Copy dependency spec first for Docker layer caching
COPY pyproject.toml .

# Install Python dependencies
# First try editable install (gets all deps from pyproject.toml)
# Then ensure fastapi[standard] is present
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir "fastapi[standard]>=0.135,<0.136"

# Copy application code
COPY . .

EXPOSE 8000

# D-03: Railway overrides via railway.toml startCommand using $PORT
# Default port 8000 for local Docker testing
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
