#!/usr/bin/env bash
# =============================================================================
# status.sh — Check vLLM VM status and health
#
# Usage: bash status.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"

echo "============================================="
echo "  GraphRAG vLLM — Status"
echo "============================================="

# --- VM status ---
echo ""
echo "==> VM:"
VM_STATUS=$(gcloud compute instances describe "${VM_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

if [ "${VM_STATUS}" = "NOT_FOUND" ]; then
    echo "    ✗ VM ${VM_NAME} not found."
    echo "    Run: bash deploy.sh"
    exit 1
fi

echo "    Name:   ${VM_NAME}"
echo "    Status: ${VM_STATUS}"

if [ "${VM_STATUS}" != "RUNNING" ]; then
    echo "    VM is not running. Start it with: bash stop.sh --start"
    exit 1
fi

# --- External IP ---
EXTERNAL_IP=$(gcloud compute instances describe "${VM_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null)

echo "    IP:     ${EXTERNAL_IP}"
echo "    URL:    http://${EXTERNAL_IP}:${VLLM_PORT}/v1"

# --- Setup progress ---
echo ""
echo "==> Setup log (last 10 lines):"
gcloud compute ssh "${VM_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --command="tail -10 /var/log/vllm-setup.log 2>/dev/null || echo '  (no log yet — startup script may still be running)'" \
    --quiet 2>/dev/null || echo "    (SSH not ready yet)"

# --- Health check ---
echo ""
echo "==> Health check:"
if curl -sf --connect-timeout 5 "http://${EXTERNAL_IP}:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "    ✓ vLLM is healthy and ready!"

    # Model info
    MODELS=$(curl -sf "http://${EXTERNAL_IP}:${VLLM_PORT}/v1/models" 2>/dev/null || echo "")
    if [ -n "${MODELS}" ]; then
        MODEL_NAME=$(echo "${MODELS}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(', '.join(m['id'] for m in d['data']))" 2>/dev/null || echo "N/A")
        echo "    Model:  ${MODEL_NAME}"
    fi

    # Quick test
    echo ""
    echo "==> Quick test:"
    RESPONSE=$(curl -sf --connect-timeout 10 "http://${EXTERNAL_IP}:${VLLM_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one sentence.\"}],
            \"max_tokens\": 50,
            \"temperature\": 0
        }" 2>/dev/null || echo "")

    if [ -n "${RESPONSE}" ]; then
        echo "    ✓ Response received:"
        echo "${RESPONSE}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
content = d['choices'][0]['message']['content']
print(f'    \"{content[:200]}\"')
" 2>/dev/null || echo "    (could not parse response)"
    else
        echo "    ✗ No response from chat endpoint"
    fi
else
    echo "    ✗ vLLM not responding yet."
    echo "    The model may still be downloading/loading (takes 5-10 min)."
    echo ""
    echo "    Check server logs:"
    echo "    gcloud compute ssh ${VM_NAME} --zone=${ZONE} --command='tail -30 /var/log/vllm-server.log'"
fi

echo ""
echo "============================================="
