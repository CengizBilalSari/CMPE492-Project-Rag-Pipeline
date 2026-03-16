#!/bin/bash

#  Load Environment (HF_TOKEN)
if [ -f .env ]; then
  export $(grep -v '^#' .env | sed 's/\r$//' | xargs -d '\n')
  echo " Environment variables loaded from .env"
else
  echo " Error: .env file not found. HF_TOKEN is required for Llama 3.1."
  exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo " Error: HF_TOKEN is not set in your .env file."
    exit 1
fi

# Configuration
STUDIO_NAME="vllm-vm"
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

echo " Deploying vLLM server with model: $MODEL_NAME"

# Remote execution
lightning studio ssh --name "$STUDIO_NAME"  << EOF

set -e

echo "📦 Activating vLLM environment..."

# Activate venv created in setup.sh
source ~/vllm-env/bin/activate

echo " Exporting HuggingFace token..."
export HF_TOKEN=$HF_TOKEN

echo " Starting vLLM server..."

nohup python -m vllm.entrypoints.openai.api_server \
  --model $MODEL_NAME \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --port 8000 \
  > ~/vllm_server.log 2>&1 &

echo "vLLM server started in background."

EOF

#  Instructions
echo "------------------------------------------------"
echo " Deployment command sent!"
echo ""
echo "First run will download the model (~15GB)."
echo "This may take several minutes."
echo ""
echo "To watch logs:"
echo ""
echo "lightning studio ssh --name $STUDIO_NAME"
echo "tail -f ~/vllm_server.log"
echo ""
