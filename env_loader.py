from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


ROOT_DIR = Path(__file__).resolve().parent
_LOADED = False


def load_project_env() -> None:
    global _LOADED
    if _LOADED or load_dotenv is None:
        return

    # Existing system environment variables keep highest priority.
    # If multiple files exist, `.env.local` wins over `.env`,
    # and `deploy.env` acts as a deployment-only fallback.
    for candidate in (ROOT_DIR / ".env.local", ROOT_DIR / ".env", ROOT_DIR / "deploy.env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)

    _LOADED = True
