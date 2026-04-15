from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Central place for runtime settings."""

    project_root: Path
    docs_dir: Path
    low_confidence_threshold: float
    top_k: int
    telegram_bot_token: str
    telegram_chat_id: str


def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[1]
    docs_dir = project_root / "data" / "docs"

    low_confidence_threshold = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.14"))
    top_k = int(os.getenv("TOP_K", "3"))

    return Settings(
        project_root=project_root,
        docs_dir=docs_dir,
        low_confidence_threshold=low_confidence_threshold,
        top_k=top_k,
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
    )
