#!/usr/bin/env bash
# =============================================================================
# connect-graphrag.sh — Update GraphRAG pipeline config to use vLLM
#
# Usage:
#   bash connect-graphrag.sh             Point pipeline at vLLM
#   bash connect-graphrag.sh --api-base http://<host>:8000/v1
#   bash connect-graphrag.sh --revert    Revert to original provider
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"

# Path to pipeline config
PIPELINE_DIR="${SCRIPT_DIR}/../graphrag_pipeline"
CONFIG_FILE="${PIPELINE_DIR}/config.yaml"
ENV_FILE="${PIPELINE_DIR}/.env"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: config.yaml not found at ${CONFIG_FILE}"
    exit 1
fi

ACTION="${1:-}"

VLLM_API_BASE_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-base)
            VLLM_API_BASE_OVERRIDE="${2:-}"
            shift 2
            ;;
        --revert|--use-openai)
            ACTION="--revert"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash connect-graphrag.sh [--api-base <url>] [--revert|--use-openai]"
            exit 1
            ;;
    esac
done

if [ "${ACTION}" = "--revert" ]; then
    # --- Revert to original ---
    echo "Reverting config.yaml to vertex provider..."

    python3 - "${CONFIG_FILE}" << 'PYEOF'
import yaml, sys

config_file = sys.argv[1]
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

config['llm']['provider'] = 'vertex'
config['llm']['model'] = 'gemini-2.5-flash'

with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print("OK: Reverted to vertex / gemini-2.5-flash")
PYEOF

else
    # --- Connect to vLLM ---
    if [ -n "${VLLM_API_BASE_OVERRIDE}" ]; then
        VLLM_API_BASE="${VLLM_API_BASE_OVERRIDE}"
    else
        EXTERNAL_IP=$(gcloud compute instances describe "${VM_NAME}" \
            --project="${PROJECT_ID}" \
            --zone="${ZONE}" \
            --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null || true)

        if [ -z "${EXTERNAL_IP}" ]; then
            echo "ERROR: Could not get VM external IP."
            echo "Either make sure the VM is running (bash status.sh),"
            echo "or pass a direct endpoint:"
            echo "  bash connect-graphrag.sh --api-base http://<host>:8000/v1"
            exit 1
        fi

        VLLM_API_BASE="http://${EXTERNAL_IP}:${VLLM_PORT}/v1"
    fi

    echo "Connecting pipeline to vLLM..."
    echo "  API base: ${VLLM_API_BASE}"
    echo "  Model:    ${MODEL}"

    # 1. Update config.yaml
    python3 - "${CONFIG_FILE}" "${MODEL}" << 'PYEOF'
import yaml, sys

config_file = sys.argv[1]
model = sys.argv[2]

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

config['llm']['provider'] = 'vllm'
config['llm']['model'] = model

with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"  OK: config.yaml updated: provider=vllm, model={model}")
PYEOF

    # 2. Set VLLM_API_BASE in .env
    if [ -f "${ENV_FILE}" ]; then
        # Remove existing VLLM_API_BASE line if present
        grep -v "^VLLM_API_BASE=" "${ENV_FILE}" > "${ENV_FILE}.tmp" || true
        mv "${ENV_FILE}.tmp" "${ENV_FILE}"
    fi
    echo "VLLM_API_BASE=${VLLM_API_BASE}" >> "${ENV_FILE}"
    echo "  OK: .env updated: VLLM_API_BASE=${VLLM_API_BASE}"

    echo ""
    echo "============================================="
    echo "  ✅ Pipeline connected to vLLM!"
    echo "============================================="
    echo ""
    echo "  To revert: bash connect-graphrag.sh --revert"
fi
