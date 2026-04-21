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

# Install vLLM
echo ""
echo "Installing vLLM..."
pip install --upgrade pip
pip install vllm

echo "  ✓ vLLM installed"

# Kill any existing vLLM process
if [ -f /tmp/vllm.pid ]; then
    OLD_PID=$(cat /tmp/vllm.pid)
    kill "${OLD_PID}" 2>/dev/null || true
    sleep 2
fi

# Start vLLM server
echo ""
echo "Starting vLLM server..."
nohup vllm serve "${MODEL}" \
    --port "${VLLM_PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --dtype "${DTYPE}" \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    > /var/log/vllm-server.log 2>&1 &

echo $! > /tmp/vllm.pid
echo "  ✓ vLLM started (PID: $(cat /tmp/vllm.pid))"

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
