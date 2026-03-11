#!/usr/bin/env bash
# =============================================================================
# budget/deploy.sh — Set up budget alert + auto-shutoff Cloud Function
#
# Usage: bash budget/deploy.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.env"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
FUNCTION_SOURCE="${SCRIPT_DIR}/function"
FUNCTION_REGION="${REGION}"

echo "============================================="
echo "  Budget Protection Setup"
echo "  Budget: ${BUDGET_AMOUNT} TRY"
echo "  Alert at: ${BUDGET_THRESHOLD} ($(echo "${BUDGET_THRESHOLD} * 100" | bc)%)"
echo "============================================="

# --- Enable APIs ---
echo ""
echo "==> [1/5] Enabling APIs..."
APIS=("cloudfunctions.googleapis.com" "pubsub.googleapis.com" "cloudbilling.googleapis.com" "cloudresourcemanager.googleapis.com" "billingbudgets.googleapis.com" "cloudbuild.googleapis.com" "run.googleapis.com" "eventarc.googleapis.com")
for api in "${APIS[@]}"; do
    STATUS=$(gcloud services list --project="${PROJECT_ID}" --filter="name:${api}" --format="value(name)" 2>/dev/null)
    if [ -z "${STATUS}" ]; then
        echo "    Enabling ${api}..."
        gcloud services enable "${api}" --project="${PROJECT_ID}" --quiet
    fi
done
echo "    APIs ready."

# --- Create Pub/Sub topic ---
echo ""
echo "==> [2/5] Creating Pub/Sub topic..."
EXISTING_TOPIC=$(gcloud pubsub topics list \
    --project="${PROJECT_ID}" \
    --filter="name:projects/${PROJECT_ID}/topics/${PUBSUB_TOPIC}" \
    --format="value(name)" 2>/dev/null)

if [ -n "${EXISTING_TOPIC}" ]; then
    echo "    Topic ${PUBSUB_TOPIC} already exists."
else
    gcloud pubsub topics create "${PUBSUB_TOPIC}" \
        --project="${PROJECT_ID}" \
        --quiet
    echo "    Topic created."
fi

# --- IAM permissions ---
echo ""
echo "==> [3/5] Setting up IAM permissions..."
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)" 2>/dev/null)
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
PUBSUB_SA="service-${PROJECT_NUMBER}@gcp-sa-pubsub.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/cloudbuild.builds.builder" \
    --quiet >/dev/null 2>&1

BILLING_ACCOUNT_ID="${BILLING_ACCOUNT_ID:-$(gcloud billing projects describe "${PROJECT_ID}" --format="value(billingAccountName)" 2>/dev/null | sed 's|billingAccounts/||')}"

gcloud billing accounts add-iam-policy-binding "${BILLING_ACCOUNT_ID}" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/billing.admin" \
    --quiet >/dev/null 2>&1

echo "    IAM permissions set."

# --- Deploy Cloud Function ---
echo ""
echo "==> [4/5] Deploying Cloud Function ${FUNCTION_NAME}..."
gcloud functions deploy "${FUNCTION_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${FUNCTION_REGION}" \
    --gen2 \
    --runtime="${FUNCTION_RUNTIME}" \
    --entry-point="${FUNCTION_ENTRY_POINT}" \
    --source="${FUNCTION_SOURCE}" \
    --trigger-topic="${PUBSUB_TOPIC}" \
    --set-env-vars="GCP_PROJECT=${PROJECT_ID}" \
    --quiet

# Grant invoke permissions
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${PUBSUB_SA}" \
    --role="roles/iam.serviceAccountTokenCreator" \
    --quiet >/dev/null 2>&1

gcloud run services add-iam-policy-binding "${FUNCTION_NAME}" \
    --region="${FUNCTION_REGION}" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/run.invoker" \
    --quiet >/dev/null 2>&1

echo "    Cloud Function deployed."

# --- Create Budget ---
echo ""
echo "==> [5/5] Creating budget..."
EXISTING_BUDGET=$(gcloud billing budgets list \
    --billing-account="${BILLING_ACCOUNT_ID}" \
    --filter="displayName=${BUDGET_DISPLAY_NAME}" \
    --format="value(name)" 2>/dev/null)

if [ -n "${EXISTING_BUDGET}" ]; then
    echo "    Budget ${BUDGET_DISPLAY_NAME} already exists."
else
    FULL_TOPIC="projects/${PROJECT_ID}/topics/${PUBSUB_TOPIC}"

    gcloud billing budgets create \
        --billing-account="${BILLING_ACCOUNT_ID}" \
        --display-name="${BUDGET_DISPLAY_NAME}" \
        --budget-amount="${BUDGET_AMOUNT}TRY" \
        --threshold-rule=percent="${BUDGET_THRESHOLD}" \
        --notifications-rule-pubsub-topic="${FULL_TOPIC}" \
        --calendar-period=month \
        --quiet

    echo "    Budget created."
fi

echo ""
echo "============================================="
echo "  ✅ Budget protection active!"
echo "============================================="
echo "  Topic:    ${PUBSUB_TOPIC}"
echo "  Function: ${FUNCTION_NAME}"
echo "  Budget:   ${BUDGET_AMOUNT} TRY (alert at $(echo "${BUDGET_THRESHOLD} * 100" | bc)%)"
echo ""
echo "  When spending exceeds the budget, billing will be"
echo "  automatically disabled on this project."
echo "============================================="
