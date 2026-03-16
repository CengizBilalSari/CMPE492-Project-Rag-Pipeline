#!/bin/bash
# ==========================================
# status.sh — SSH into Lightning VM and tail vLLM logs with .env credentials
# ==========================================

# Load environment variables from .env

if [ -f .env ]; then
  export $(grep -v '^#' .env | sed 's/\r$//' | xargs -d '\n')
  echo "✅ Environment variables loaded."

  echo "--- Variable Verification ---"
  if [ -n "$LIGHTNING_USER_ID" ]; then
    echo "User ID: ${LIGHTNING_USER_ID:0:4}****"
  else
    echo " User ID NOT FOUND in .env"
  fi

  if [ -n "$LIGHTNING_API_KEY" ]; then
    echo "API Key: ${LIGHTNING_API_KEY:0:4}****"
  else
    echo " API Key NOT FOUND in .env"
  fi
  echo "----------------------------"
else
  echo " Error: .env file not found."
  exit 1
fi

# Replace with your Lightning VM name
VM_NAME="vllm-vm"

# Path to the log file on the VM
LOG_FILE="~/vllm_server.log"

echo "📡 Connecting to VM '$VM_NAME' and tailing logs..."

# SSH with heredoc and exported environment variables
lightning studio ssh --name "$VM_NAME" << EOF
echo "🔹 Connected to VM. Tailing vLLM logs..."
tail -n 20 -f $LOG_FILE
EOF