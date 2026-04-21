import httpx
import json
import os
import re
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Configuration from .env or defaults
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
DEFAULT_MODEL = "deepseek/deepseek-r1-0528-qwen3-8b"

def call_llm_studio(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = -1,
    top_p: float = 0.7,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    stop: Optional[List[str]] = None,
    stream: bool = False
) -> str:
    """
    Sends a request to LM Studio API with parameters similar to the interface.
    
    Args:
        prompt: The user prompt to send.
        system_prompt: The system message to set context.
        model: Model identifier (from LM Studio).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate (-1 for no limit/model default).
        top_p: Nucleus sampling probability.
        presence_penalty: Penalty for repeating tokens currently in prompt.
        frequency_penalty: Penalty for repeating tokens based on their frequency.
        stop: List of stop sequences.
        stream: Whether to stream the response (currently returns full text).
        
    Returns:
        The generated content as a string.
    """
    
    # Ensure URL is correct
    base_url = LMSTUDIO_BASE_URL.replace("/chat/completions", "").rstrip("/")
    endpoint = f"{base_url}/chat/completions"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens if max_tokens > 0 else 2048,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "stop": stop or [],
        "stream": stream
    }
    
    print(f"\n[LM-Studio] Calling model: {model}")
    print(f"[LM-Studio] Endpoint: {endpoint}")
    print(f"[LM-Studio] Parameters: temp={temperature}, top_p={top_p}, max_tokens={payload['max_tokens']}")
    
    start_time = time.time()
    
    try:
        with httpx.Client(timeout=300.0) as client:
            response = client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            
            elapsed = time.time() - start_time
            content = data["choices"][0]["message"]["content"] or ""
            
            # Post-processing: Remove <think> tags if present (common in DeepSeek)
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            
            usage = data.get("usage", {})
            print(f"[LM-Studio] Completed in {elapsed:.2f}s | Tokens: {usage.get('total_tokens', 'N/A')}")
            
            return content
            
    except httpx.HTTPStatusError as e:
        return f"HTTP Error: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error connecting to LM Studio: {str(e)}"

if __name__ == "__main__":
    # Example usage
    sample_prompt = "Compare GraphRAG with traditional Vector RAG in 3 bullet points."
    
    # You can change these parameters just like in the LM Studio interface
    result = call_llm_studio(
        prompt=sample_prompt,
        system_prompt="You are an expert AI researcher.",
        temperature=0.7,
        top_p=0.9,
        max_tokens=1000
    )
    
    print("\n--- Model Response ---\n")
    print(result)
