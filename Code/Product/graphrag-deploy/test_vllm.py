import os
import asyncio
from openai import AsyncOpenAI

# 1. Get the VM's external IP from gcloud (or just hardcode it here)
# For example: VLLM_API_BASE = "http://35.242.232.115:8000/v1"
api_base = os.getenv("VLLM_API_BASE")

if not api_base:
    print("WARNING: VLLM_API_BASE environment variable not found.")
    print("Please set your VM IP manually in the script, or run:")
    print("  export VLLM_API_BASE=http://<VM_IP>:8000/v1")
    exit(1)

print(f"Connecting to vLLM at: {api_base}")

# 2. We use the exact same OpenAI client, just pointing to our VM
client = AsyncOpenAI(
    api_key="dummy_key_not_needed_for_vllm", 
    base_url=api_base,
    max_retries=0
)

async def test_vllm():
    try:
        # 3. Create a chat completion request
        print("\nSending prompt: 'Explain what a Graph Database is in 2 short sentences.'...")
        
        response = await client.chat.completions.create(
            # Must match the MODEL in config.env
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Explain what a Graph Database is in 2 short sentences."}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        # 4. Print the result
        print("\n=== SUCCESS: Response Received ===")
        print(response.choices[0].message.content)
        print("==================================")
        
        # Print token usage stats
        print(f"\nUsage Stats:")
        print(f"  Prompt tokens:     {response.usage.prompt_tokens}")
        print(f"  Completion tokens: {response.usage.completion_tokens}")
        print(f"  Total tokens:      {response.usage.total_tokens}")

    except Exception as e:
        print(f"\n=== ERROR ===")
        print(f"Failed to connect to vLLM: {e}")
        print("Make sure your VM is running and port 8000 is open.")

if __name__ == "__main__":
    asyncio.run(test_vllm())
