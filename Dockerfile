# Base image with Python 3.11/3.12
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose ports for both FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command launches FastAPI microservice
CMD ["uvicorn", "web_app.api:app", "--host", "0.0.0.0", "--port", "8000"]
