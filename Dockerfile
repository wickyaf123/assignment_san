FROM python:3.12-slim

# Install Java 17 for OpenDataLoader PDF extraction (D-02)
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME explicitly (Pitfall 1 from RESEARCH.md)
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
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
