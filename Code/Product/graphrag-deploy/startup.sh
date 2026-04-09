#!/bin/bash
# =============================================================================
# startup.sh — Runs on the GCE VM at boot to install and start vLLM
#
# Config is read from VM metadata (set by deploy.sh)
# =============================================================================
set -e

LOG_FILE="/var/log/vllm-setup.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========== vLLM Setup Starting =========="
echo "Time: $(date)"

# Get config from VM metadata
META_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
META_HEADER="Metadata-Flavor: Google"

HF_TOKEN=$(curl -sf -H "${META_HEADER}" "${META_URL}/hf-token")
MODEL=$(curl -sf -H "${META_HEADER}" "${META_URL}/vllm-model")
MAX_MODEL_LEN=$(curl -sf -H "${META_HEADER}" "${META_URL}/max-model-len")
GPU_MEM_UTIL=$(curl -sf -H "${META_HEADER}" "${META_URL}/gpu-mem-util")
DTYPE=$(curl -sf -H "${META_HEADER}" "${META_URL}/dtype")
VLLM_PORT=$(curl -sf -H "${META_HEADER}" "${META_URL}/vllm-port")

export HF_TOKEN

echo "Config loaded from metadata:"
echo "  Model:           ${MODEL}"
echo "  Max model len:   ${MAX_MODEL_LEN}"
echo "  GPU mem util:    ${GPU_MEM_UTIL}"
echo "  DType:           ${DTYPE}"
echo "  Port:            ${VLLM_PORT}"

# Wait for GPU driver
echo ""
echo "Waiting for GPU driver..."
for i in $(seq 1 60); do
    if nvidia-smi > /dev/null 2>&1; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "unknown")
        echo "  ✓ GPU ready: ${GPU_NAME} (${GPU_MEM})"
        break
    fi
    echo "  Attempt ${i}/60 — GPU not ready yet, waiting 10s..."
    sleep 10
done

if ! nvidia-smi > /dev/null 2>&1; then
    echo "  ✗ GPU driver failed to load after 10 minutes!"
    exit 1
fi

# Install Docker (if not already present)
echo ""
echo "Setting up Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    echo "  ✓ Docker installed"
else
    echo "  ✓ Docker already installed"
fi

# Ensure NVIDIA Container Toolkit is available
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update && apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    echo "  ✓ NVIDIA Container Toolkit installed"
else
    echo "  ✓ NVIDIA Container Toolkit already installed"
fi

echo "  ✓ Docker ready"

# Kill any existing vLLM container
echo ""
echo "Cleaning up old containers..."
docker rm -f vllm-server 2>/dev/null || true

# Start vLLM server via Docker
echo ""
echo "Starting vLLM server (Docker)..."
docker run -d \
    --name vllm-server \
    --gpus all \
    -p "${VLLM_PORT}:${VLLM_PORT}" \
    -e "HF_TOKEN=${HF_TOKEN}" \
    --shm-size=4g \
    vllm/vllm-openai:latest \
    --model "${MODEL}" \
    --port "${VLLM_PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --dtype "${DTYPE}" \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --enable-prefix-caching \
    --max-num-seqs 32

echo "  ✓ vLLM container started"

# Wait for health
echo ""
echo "Waiting for vLLM to become healthy..."
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "  ✓ vLLM is healthy and ready!"
        echo ""
        echo "========== vLLM Setup Complete =========="
        exit 0
    fi
    if [ $((i % 12)) -eq 0 ]; then
        echo "  Still loading... ($((i * 5))s elapsed)"
    fi
    sleep 5
done

echo "  ⚠ vLLM did not become healthy within 10 minutes."
echo "  Check: tail -50 /var/log/vllm-server.log"
echo "========== vLLM Setup Complete (with warning) =========="
