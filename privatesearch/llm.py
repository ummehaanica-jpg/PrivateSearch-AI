"""Local LLM access through Ollama. Requests go to localhost only (enforced)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import requests

from .config import is_loopback_url


class LLMError(Exception):
    """A problem talking to the local model, with a message safe to show to the user."""


class LLMClient(Protocol):
    model: str

    def generate(self, system: str, prompt: str) -> str: ...


@dataclass
class OllamaStatus:
    running: bool
    models: list[str] = field(default_factory=list)
    has_model: bool = False
    message: str = ""


class OllamaClient:
    def __init__(self, host: str, model: str, timeout: int = 240, allow_remote: bool = False):
        host = (host or "").rstrip("/")
        if not host.startswith(("http://", "https://")):
            raise LLMError("The Ollama address must start with http:// or https://")
        if not allow_remote and not is_loopback_url(host):
            raise LLMError(
                "Privacy protection: only a model running on this computer (localhost) is allowed. "
                "Set PSAI_ALLOW_REMOTE_LLM=1 to override."
            )
        self.host, self.model, self.timeout = host, model, timeout

    def status(self) -> OllamaStatus:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=3)
            response.raise_for_status()
            names = [m.get("name", "") for m in response.json().get("models", [])]
        except (requests.RequestException, ValueError):
            return OllamaStatus(False, message="Ollama is not running. Start it by opening the Ollama app.")
        wanted = {self.model, f"{self.model}:latest"}
        has = any(n in wanted or n.split(":")[0] == self.model for n in names)
        message = "Ready." if has else f"Model '{self.model}' is not installed. Run: ollama pull {self.model}"
        return OllamaStatus(True, names, has, message)

    def generate(self, system: str, prompt: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "options": {"temperature": 0.1, "num_ctx": 4096},
        }
        try:
            response = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise LLMError("Could not reach Ollama. Is it running?") from exc
        except requests.Timeout as exc:
            raise LLMError("The local model took too long to answer. Try a smaller model.") from exc
        if response.status_code == 404:
            raise LLMError(f"Model '{self.model}' is not installed. Run: ollama pull {self.model}")
        if not response.ok:
            raise LLMError(f"Ollama returned an error ({response.status_code}).")
        try:
            return str(response.json()["message"]["content"]).strip()
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError("Ollama sent a response that could not be read.") from exc
