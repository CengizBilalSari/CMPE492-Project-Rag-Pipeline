# GraphRAG vLLM Deployment (GCE VM)

Deploy a self-hosted LLM on a single GCP VM for your GraphRAG project.

## Directory Structure

```
graphrag-deploy/
├── config.env              # All configuration (GCP, VM, vLLM, budget)
├── setup.sh                # Prerequisites (gcloud, APIs, GPU quota)
├── deploy.sh               # Create VM with GPU + start vLLM
├── startup.sh              # Runs on the VM: install vLLM, start serving
├── status.sh               # Health check + quick test
├── stop.sh                 # Stop / start / destroy VM
├── connect-graphrag.sh     # Update GraphRAG settings.yaml
├── budget/
│   ├── deploy.sh           # Budget alert + auto-shutoff
│   └── function/
│       ├── main.py         # Cloud Function: disable billing
│       └── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Set your GCP project
gcloud config set project <YOUR_PROJECT_ID>

# 2. Check prerequisites (GPU quota, APIs)
bash setup.sh

# 3. (Optional) Set up budget protection
bash budget/deploy.sh

# 4. Deploy vLLM on a GPU VM
export HF_TOKEN="hf_xxxxx"
bash deploy.sh              # ~2 min to create VM, ~5-10 min for vLLM to start

# 5. Check if vLLM is ready
bash status.sh

# 6. Connect GraphRAG to vLLM
bash connect-graphrag.sh

# 7. Use GraphRAG
cd ../grapgrag-exp
graphrag query --method global --query "What is this dataset about?"
```

## Cost Management

| Command | Effect | Cost |
|---|---|---|
| Running | VM + GPU active | ~$1.40/hr |
| `bash stop.sh --stop` | VM stopped (disk retained) | ~$0.13/day |
| `bash stop.sh --start` | Resume stopped VM | ~$1.40/hr |
| `bash stop.sh --destroy` | Delete everything | $0 |

**No hidden GKE management fees** — just the VM and disk.

## Reverting to OpenAI

```bash
bash connect-graphrag.sh --use-openai
```
