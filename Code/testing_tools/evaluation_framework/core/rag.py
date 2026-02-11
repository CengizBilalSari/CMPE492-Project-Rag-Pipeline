from typing import Dict, List

try:
    import requests
except Exception:  
    requests = None


class RAGClient:
    def __init__(self, endpoint_url: str, http_method: str = "post", top_k: int = 5):
        if requests is None:
            raise ImportError("requests is not installed. Install with `pip install requests`.")
        self.endpoint_url = endpoint_url
        self.http_method = http_method.lower()
        self.top_k = top_k

    def retrieve(self, query: str) -> Dict:
        payload = {"query": query, "top_k": self.top_k}
        if self.http_method == "get":
            resp = requests.get(self.endpoint_url, params=payload, timeout=30)
        else:
            resp = requests.post(self.endpoint_url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Extract contexts
        contexts: List[str] = []
        if isinstance(data, dict):
            if "contexts" in data:
                contexts = data["contexts"]
            elif "documents" in data:
                contexts = data["documents"]
            elif "results" in data:
                contexts = data["results"]
        elif isinstance(data, list):
            contexts = data

        answer = ""
        if isinstance(data, dict) and "answer" in data:
            answer = data["answer"]
        elif contexts:
            answer = contexts[0]

        token_usage = {}
        if isinstance(data, dict) and "token_usage" in data:
            token_usage = data["token_usage"]

        return {
            "answer": answer,
            "retrieved_contexts": contexts,
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0),
        }
