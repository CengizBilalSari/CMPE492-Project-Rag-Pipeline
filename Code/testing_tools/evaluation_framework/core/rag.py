from typing import List

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

    def retrieve(self, query: str) -> List[str]:
        payload = {"query": query, "top_k": self.top_k}
        if self.http_method == "get":
            resp = requests.get(self.endpoint_url, params=payload, timeout=30)
        else:
            resp = requests.post(self.endpoint_url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            if "contexts" in data:
                return data["contexts"]
            if "documents" in data:
                return data["documents"]
            if "results" in data:
                return data["results"]
        if isinstance(data, list):
            return data
        return []
