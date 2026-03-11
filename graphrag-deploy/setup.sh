#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-time prerequisites for GCP deployment
#
# Usage: bash setup.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
if [ -z "${PROJECT_ID}" ]; then
    echo "ERROR: No GCP project set. Run: gcloud config set project <PROJECT_ID>"
    exit 1
fi

echo "============================================="
echo "  GraphRAG vLLM — Prerequisites Setup"
echo "  Project: ${PROJECT_ID}"
echo "============================================="

# --- Check required tools ---
echo ""
echo "==> [1/3] Checking required tools..."
if ! command -v gcloud >/dev/null 2>&1; then
    echo "ERROR: gcloud not found."
    echo "Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi
echo "    gcloud — OK"

# --- Enable required APIs ---
echo ""
echo "==> [2/3] Enabling GCP APIs..."
APIS=("compute.googleapis.com")
for api in "${APIS[@]}"; do
    STATUS=$(gcloud services list --project="${PROJECT_ID}" --filter="name:${api}" --format="value(name)" 2>/dev/null)
    if [ -z "${STATUS}" ]; then
        echo "    Enabling ${api}..."
        gcloud services enable "${api}" --project="${PROJECT_ID}" --quiet
    else
        echo "    ${api} — already enabled"
    fi
done

# --- Check GPU quota ---
echo ""
echo "==> [3/3] Checking GPU quota..."
GPU_QUOTA=$(gcloud compute project-info describe \
    --project="${PROJECT_ID}" \
    --format="value(quotas[name=GPUS_ALL_REGIONS].limit)" 2>/dev/null || echo "0")

if [ -z "${GPU_QUOTA}" ] || [ "${GPU_QUOTA}" = "0" ]; then
    echo ""
    echo "    ✗ GPU quota (GPUS_ALL_REGIONS) is ${GPU_QUOTA:-0}."
    echo "    You need at least 1 GPU."
    echo ""
    echo "    Request a quota increase at:"
    echo "    https://console.cloud.google.com/iam-admin/quotas?project=${PROJECT_ID}"
    echo ""
    echo "    Steps:"
    echo "      1. Filter for 'GPUs (all regions)'"
    echo "      2. Select it → Edit Quotas → set limit to 1"
    echo "      3. Wait for approval (usually instant to a few hours)"
    echo ""
    echo "    After approved, re-run this script."
    exit 1
else
    echo "    GPU quota: ${GPU_QUOTA} — OK"
fi

echo ""
echo "============================================="
echo "  ✅ Prerequisites OK!"
echo "============================================="
echo ""
echo "  Next steps:"
echo "    1. bash deploy.sh"
echo ""
