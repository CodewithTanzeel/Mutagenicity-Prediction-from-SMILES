FROM python:3.11-slim

# 1. Install the hidden Linux libraries RDKit needs to draw images
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy your app code
COPY . .