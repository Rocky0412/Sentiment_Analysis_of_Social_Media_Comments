FROM python:3.10-slim

WORKDIR /app

# Copy application code
COPY flask-app/ /app/

# Copy model metadata correctly
COPY Model_info/ /app/Model_info/

# System deps (important for sklearn, scipy)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt \
    boto3 \
    botocore

EXPOSE 5000

CMD ["python", "app.py"]

