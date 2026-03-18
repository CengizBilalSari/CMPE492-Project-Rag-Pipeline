# Use Python 3.12.3 as base image
FROM python:3.12.3-slim

# Set working directory
WORKDIR /app/lightning-ai

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openssh-client \
    grep \
    sed \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to 24.0
RUN python -m pip install --upgrade pip==24.0

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts and python files
COPY . .

# Ensure scripts are executable
RUN chmod +x *.sh

# Support for interactive mode and automatic line-ending fixes for Windows mounts
CMD ["sh", "-c", "sed -i 's/\\r$//' *.sh && exec bash"]
