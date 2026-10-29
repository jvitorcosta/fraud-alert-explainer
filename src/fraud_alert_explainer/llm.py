"""Minimal Ollama chat client (local model, temperature 0)."""

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:11434"


class OllamaClient:
    def __init__(self, http: httpx.Client, model: str) -> None:
        self._http = http
        self.model = model

    def chat(self, system: str, user: str, *, max_tokens: int = 200) -> str:
        """One non-streamed reply."""
        response = self._http.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": 0, "num_predict": max_tokens, "seed": 0},
            },
        )
        response.raise_for_status()
        return str(response.json()["message"]["content"]).strip()
