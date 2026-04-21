#!/bin/bash


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

STUDIO_NAME="vllm-vm"

# CHECK STUDIO EXISTENCE
echo "Checking if Studio '$STUDIO_NAME' exists..."
# -i (case insensitive), -q (quiet), -w (whole word)
if ! lightning studio list | grep -i -q -w "$STUDIO_NAME"; then
    echo "Studio '$STUDIO_NAME' not found. Creating it..."
    lightning studio create --name "$STUDIO_NAME"
else
    echo "✅ Studio '$STUDIO_NAME' found."
fi

# START STUDIO
echo "Starting $STUDIO_NAME with L4 GPU..."
lightning studio start --name "$STUDIO_NAME" --machine L4

# POLLING LOOP (Wait for 'Running')
echo "Waiting for VM to be ready..."
MAX_RETRIES=30
COUNT=0

while true; do
    # Capture the line for our studio from the list
    CURRENT_LINE=$(lightning studio list | grep -w "$STUDIO_NAME")
    
    # Check if the word 'Running' appears anywhere in that line
    if echo "$CURRENT_LINE" | grep -q "Running"; then
        echo "✅ VM is now Running!"
        break
    fi

    if [ $COUNT -eq $MAX_RETRIES ]; then
        echo "❌ Timeout: Studio took too long to start."
        exit 1
    fi

    echo "Status: Starting... ($((COUNT * 10))s)"
    sleep 10
    ((COUNT++))
done

# REMOTE vLLM INSTALLATION

echo " Installing/Updating vLLM on remote VM..."
lightning studio ssh --name "$STUDIO_NAME" << 'EOF'
set -e

echo "Installing python3-pip..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

echo "Creating venv..."
python3 -m venv ~/vllm-env

echo "Activating venv..."
source ~/vllm-env/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing vLLM..."
pip install vllm

echo "vLLM installation finished."
EOF

echo "------------------------------------------------"
echo " Setup complete! You can now access your VM via:"
echo "lightning studio ssh --name $STUDIO_NAME"