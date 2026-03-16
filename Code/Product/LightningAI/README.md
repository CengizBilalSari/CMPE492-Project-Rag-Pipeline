# LightningAI vLLM Deployment

This directory contains scripts to provision a VM on [Lightning AI](https://lightning.ai/), install [vLLM](https://github.com/vllm-project/vllm), and deploy large language models (like Mistral or Llama) as an OpenAI-compatible API.

## Prerequisites

-   A **Lightning AI** account.
-   Hugging Face Token (`HF_TOKEN`) for gated models (like Llama 3.1).
-   Lightning CLI installed locally (included in `requirements.txt`).

## Getting Started (Order of Operations)

Follow these steps in order to set up your environment and deploy the model.

### 1. Local Environment Setup

First, set up a local Python environment to manage the Lightning CLI and environment variables.

```bash
# Create a virtual environment (Unix/WSL/macOS)
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install required local dependencies
pip install -r requirements.txt
```

### 2. Configuration

Set up your credentials by copying the example environment file.

```bash
cp .env.example .env
```

Edit the `.env` file and fill in the following:
-   `LIGHTNING_USER_ID`: Your Lightning AI User ID.
-   `LIGHTNING_API_KEY`: Your Lightning AI API Key.
-   `HF_TOKEN`: Your Hugging Face access token.

### 3. Provisioning the VM

Run the setup script to create a Lightning Studio (VM) and install vLLM on it. This script uses an `L4` GPU by default.

```bash
chmod +x *.sh
./setup.sh
```

### 4. Deploying the Model

Once the VM is ready, deploy the vLLM server with your chosen model. The default model is `mistralai/Mistral-7B-Instruct-v0.3`. The llama 3 models needs a request to the huggingface to download the model weights. (They approved mine in less than a 10 minute).

```bash
./deploy.sh
```
> [!NOTE]
> The first run will download the model weights, which may take several minutes.

### 5. Monitoring Status

To check if the server is running or to troubleshoot the download progress, use the status script to tail the remote logs:

```bash
./status.sh
```

## Open Lightning Studio on Browser
- Click the Port Viewer on the right, create port 8000, it gives you public url.
- Now you can use this url on vllm_test.py

## Destroy the VM
- After you are done, you can destroy the VM with the destroy script.
```bash
./destroy.sh
```

## File Overview

-   `setup.sh`: Creates the "vllm-vm" Studio, starts it with an L4 GPU, and installs vLLM in a remote venv.
-   `deploy.sh`: Activates the remote venv and starts the `vllm.entrypoints.openai.api_server`.
-   `status.sh`: Connects via SSH and tails `~/vllm_server.log`.
-   `vllm_test.py`: A simple script to verify the API connection.
-   `requirements.txt`: Contains local dependencies (lightning, python-dotenv, etc.).
