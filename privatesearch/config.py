"""Configuration and privacy settings.

Everything is configurable through environment variables (or a local `.env`
file) and nothing is hard-coded to an absolute path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALLOWED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md", ".markdown")
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env reader (KEY=VALUE per line). Existing variables win."""
    path = path or PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_loopback_url(url: str) -> bool:
    """True when the URL points at this computer."""
    try:
        host = urlparse(url).hostname
    except ValueError:
        return False
    return (host or "").lower() in LOOPBACK_HOSTS


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedder_backend: str = "auto"  # auto | sentence-transformers | hash
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    chunk_words: int = 160
    chunk_overlap_words: int = 30
    max_upload_mb: int = 50
    max_pdf_pages: int = 2000
    allow_remote_llm: bool = False
    offline_mode: bool = False

    @property
    def db_path(self) -> Path:
        return self.data_dir / "privatesearch.db"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, data_dir: str | Path | None = None) -> "Settings":
        load_dotenv()
        chosen = data_dir or os.environ.get("PSAI_DATA_DIR") or PROJECT_ROOT / "data"
        data_path = Path(chosen)
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path
        return cls(
            data_dir=data_path,
            embedding_model=os.environ.get("PSAI_EMBEDDING_MODEL", cls.embedding_model),
            embedder_backend=os.environ.get("PSAI_EMBEDDER", "auto").lower(),
            ollama_host=os.environ.get("PSAI_OLLAMA_HOST", cls.ollama_host),
            ollama_model=os.environ.get("PSAI_OLLAMA_MODEL", cls.ollama_model),
            chunk_words=_env_int("PSAI_CHUNK_WORDS", cls.chunk_words),
            chunk_overlap_words=_env_int("PSAI_CHUNK_OVERLAP", cls.chunk_overlap_words),
            max_upload_mb=_env_int("PSAI_MAX_UPLOAD_MB", cls.max_upload_mb),
            max_pdf_pages=_env_int("PSAI_MAX_PDF_PAGES", cls.max_pdf_pages),
            allow_remote_llm=_env_bool("PSAI_ALLOW_REMOTE_LLM", False),
            offline_mode=_env_bool("PSAI_OFFLINE", False),
        )


def apply_privacy_environment(settings: Settings) -> None:
    """Switch off telemetry in the libraries we use. Call before importing ML libs."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if settings.offline_mode:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
