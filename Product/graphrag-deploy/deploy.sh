#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Create a GCE VM with GPU and deploy vLLM
#
# Usage:
#   export HF_TOKEN="hf_xxxxx"
#   bash deploy.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"

# --- Validate HF_TOKEN ---
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set."
    echo "Run: export HF_TOKEN=\"hf_xxxxx\""
    exit 1
fi

echo "============================================="
echo "  GraphRAG vLLM — VM Deployment"
echo "  Project: ${PROJECT_ID}"
echo "  Zone:    ${ZONE}"
echo "  VM:      ${VM_NAME} (${MACHINE_TYPE})"
echo "  Model:   ${MODEL}"
echo "============================================="
echo ""

# =========================================================================
# 1. CREATE FIREWALL RULE
# =========================================================================
EXISTING_FW=$(gcloud compute firewall-rules list \
    --project="${PROJECT_ID}" \
    --filter="name=${FIREWALL_RULE}" \
    --format="value(name)" 2>/dev/null)

if [ -n "${EXISTING_FW}" ]; then
    echo "==> [1/2] Firewall rule ${FIREWALL_RULE} already exists."
else
    echo "==> [1/2] Creating firewall rule to allow port ${VLLM_PORT}..."
    gcloud compute firewall-rules create "${FIREWALL_RULE}" \
        --project="${PROJECT_ID}" \
        --allow="tcp:${VLLM_PORT}" \
        --target-tags="vllm-server" \
        --description="Allow access to vLLM server"
    echo "    Firewall rule created."
fi

# =========================================================================
# 2. CREATE VM
# =========================================================================
EXISTING_VM=$(gcloud compute instances list \
    --project="${PROJECT_ID}" \
    --zones="${ZONE}" \
    --filter="name=${VM_NAME}" \
    --format="value(name)" 2>/dev/null)

if [ -n "${EXISTING_VM}" ]; then
    echo "==> [2/2] VM ${VM_NAME} already exists."
    VM_STATUS=$(gcloud compute instances describe "${VM_NAME}" \
        --project="${PROJECT_ID}" \
        --zone="${ZONE}" \
        --format="value(status)" 2>/dev/null)

    if [ "${VM_STATUS}" = "TERMINATED" ]; then
        echo "    VM is stopped. Starting it..."
        gcloud compute instances start "${VM_NAME}" \
            --project="${PROJECT_ID}" \
            --zone="${ZONE}"
        echo "    VM started."
    else
        echo "    VM status: ${VM_STATUS}"
    fi
else
    echo "==> [2/2] Creating VM ${VM_NAME}..."
    gcloud compute instances create "${VM_NAME}" \
        --project="${PROJECT_ID}" \
        --zone="${ZONE}" \
        --machine-type="${MACHINE_TYPE}" \
        --accelerator="type=nvidia-l4,count=1" \
        --maintenance-policy=TERMINATE \
        --image-family="${IMAGE_FAMILY}" \
        --image-project="${IMAGE_PROJECT}" \
        --boot-disk-size="${DISK_SIZE_GB}GB" \
        --tags="vllm-server" \
        --metadata="hf-token=${HF_TOKEN},vllm-model=${MODEL},max-model-len=${MAX_MODEL_LEN},gpu-mem-util=${GPU_MEMORY_UTILIZATION},dtype=${DTYPE},vllm-port=${VLLM_PORT},install-nvidia-driver=True" \
        --metadata-from-file="startup-script=${SCRIPT_DIR}/startup.sh" \
        --scopes="default"

    echo "    VM created."
fi

# =========================================================================
# GET EXTERNAL IP
# =========================================================================
echo ""
echo "Getting external IP..."
EXTERNAL_IP=""
for i in $(seq 1 12); do
    EXTERNAL_IP=$(gcloud compute instances describe "${VM_NAME}" \
        --project="${PROJECT_ID}" \
        --zone="${ZONE}" \
        --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null)
    if [ -n "${EXTERNAL_IP}" ]; then
        break
    fi
    sleep 5
done

echo ""
if [ -n "${EXTERNAL_IP}" ]; then
    echo "============================================="
    echo "  ✅ VM created!"
    echo "============================================="
    echo ""
    echo "  External IP : ${EXTERNAL_IP}"
    echo "  vLLM URL    : http://${EXTERNAL_IP}:${VLLM_PORT}"
    echo "  API base    : http://${EXTERNAL_IP}:${VLLM_PORT}/v1"
    echo ""
    echo "  ⏳ vLLM is installing and loading the model."
    echo "  This takes 5-10 minutes on first boot."
    echo ""
    echo "  Check progress:"
    echo "    bash status.sh"
    echo ""
    echo "  Watch live logs:"
    echo "    gcloud compute ssh ${VM_NAME} --zone=${ZONE} --command='tail -f /var/log/vllm-setup.log'"
    echo ""
else
    echo "  ⚠  Could not get external IP."
    echo "  Check: gcloud compute instances describe ${VM_NAME} --zone=${ZONE}"
fi
