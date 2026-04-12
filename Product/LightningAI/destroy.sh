#!/bin/bash

# WARNING: This will permanently delete your studio. Data cannot be recovered.
STUDIO_NAME="vllm-vm"

echo "🚨 Initiating teardown for: $STUDIO_NAME"

# Check if the studio exists in the list
if lightning studio list | grep -q "$STUDIO_NAME"; then
    echo "💥 Deleting studio $STUDIO_NAME..."
    
    lightning delete studio "$STUDIO_NAME"
    
    echo "✅ Studio $STUDIO_NAME has been queued for deletion."
else
    echo "ℹ️ Studio $STUDIO_NAME not found in your account."
fi