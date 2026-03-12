#!/usr/bin/env bash
# =============================================================================
# stop.sh — Stop, start, or destroy the vLLM VM
#
# Usage:
#   bash stop.sh --stop      Stop the VM (no GPU charges, disk charges only)
#   bash stop.sh --start     Start a stopped VM
#   bash stop.sh --destroy   Delete VM + firewall (zero cost)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"

usage() {
    echo "Usage: bash stop.sh [--stop | --start | --destroy]"
    echo ""
    echo "  --stop     Stop the VM (GPU charges stop, ~\$0.13/day disk cost remains)"
    echo "  --start    Start a stopped VM (vLLM auto-starts via startup script)"
    echo "  --destroy  Delete VM + firewall + orphaned disks (zero cost)"
    exit 1
}

cmd_stop() {
    echo "Stopping VM ${VM_NAME}..."
    echo "    [DEBUG] Sending stop command to VM ${VM_NAME} in zone ${ZONE}..."
    gcloud compute instances stop "${VM_NAME}" \
        --project="${PROJECT_ID}" \
        --zone="${ZONE}"

    echo ""
    echo "✓ VM stopped. GPU charges have stopped."
    echo "  Disk charges (~\$0.13/day for ${DISK_SIZE_GB}GB) still apply."
    echo ""
    echo "  To resume:  bash stop.sh --start"
    echo "  To destroy: bash stop.sh --destroy"
}

cmd_start() {
    echo "Starting VM ${VM_NAME}..."
    echo "    [DEBUG] Sending start command to VM ${VM_NAME} in zone ${ZONE}..."
    gcloud compute instances start "${VM_NAME}" \
        --project="${PROJECT_ID}" \
        --zone="${ZONE}"

    EXTERNAL_IP=$(gcloud compute instances describe "${VM_NAME}" \
        --project="${PROJECT_ID}" \
        --zone="${ZONE}" \
        --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null)

    echo ""
    echo "✓ VM started."
    echo "  IP:  ${EXTERNAL_IP}"
    echo "  URL: http://${EXTERNAL_IP}:${VLLM_PORT}/v1"
    echo ""
    echo "  ⏳ vLLM will auto-start. Wait 5-10 min then check:"
    echo "     bash status.sh"
}

cmd_destroy() {
    echo "============================================="
    echo "  Destroying all resources"
    echo "============================================="

    # Delete VM
    echo ""
    echo "Deleting VM ${VM_NAME}..."
    echo "    [DEBUG] Deleting VM ${VM_NAME} along with all attached disks..."
    gcloud compute instances delete "${VM_NAME}" \
        --project="${PROJECT_ID}" \
        --zone="${ZONE}" \
        --delete-disks=all || echo "  VM not found or already deleted."

    # Delete firewall rule
    echo ""
    echo "Deleting firewall rule ${FIREWALL_RULE}..."
    echo "    [DEBUG] Deleting firewall rule ${FIREWALL_RULE}..."
    gcloud compute firewall-rules delete "${FIREWALL_RULE}" \
        --project="${PROJECT_ID}" || echo "  Firewall rule not found or already deleted."

    # Check for orphaned disks
    echo ""
    echo "Checking for orphaned disks..."
    ORPHAN_DISKS=$(gcloud compute disks list \
        --project="${PROJECT_ID}" \
        --filter="zone:${ZONE} AND -users:*" \
        --format="value(name)" 2>/dev/null)

    if [ -n "${ORPHAN_DISKS}" ]; then
        echo "${ORPHAN_DISKS}" | while read -r disk; do
            echo "    [DEBUG] Deleting orphaned disk: ${disk}"
            gcloud compute disks delete "${disk}" \
                --project="${PROJECT_ID}" \
                --zone="${ZONE}" || true
        done
    else
        echo "  No orphaned disks."
    fi

    echo ""
    echo "============================================="
    echo "  ✓ All resources destroyed. Cost is now \$0."
    echo "============================================="
    echo ""
    echo "  To redeploy: bash deploy.sh"
}

if [ $# -eq 0 ]; then
    usage
fi

case "$1" in
    --stop)    cmd_stop ;;
    --start)   cmd_start ;;
    --destroy) cmd_destroy ;;
    *)         usage ;;
esac
