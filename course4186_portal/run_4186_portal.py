import os
from pathlib import Path
import sys


PORTAL_DIR = Path(__file__).resolve().parent
ROOT_DIR = PORTAL_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from kimi_utils import KIMI_API_BASE, KIMI_API_KEY, KIMI_MODEL
except Exception:
    KIMI_API_BASE = None
    KIMI_API_KEY = None
    KIMI_MODEL = None

preferred_artifact_dir = ROOT_DIR / "course4186_rag" / "artifacts_week1_week6"
if preferred_artifact_dir.exists():
    os.environ.setdefault("COURSE4186_ARTIFACT_DIR", str(preferred_artifact_dir))

if not any(os.getenv(name) for name in ("COURSE_LLM_API_KEY", "DEEPSEEK_API_KEY", "KIMI_API_KEY")) and KIMI_API_KEY:
    os.environ.setdefault("COURSE_LLM_API_KEY", KIMI_API_KEY)
    if KIMI_API_BASE:
        os.environ.setdefault("COURSE_LLM_BASE_URL", KIMI_API_BASE)
    if KIMI_MODEL:
        os.environ.setdefault("COURSE_LLM_MODEL", KIMI_MODEL)

from course4186_portal.app import create_app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=50186, debug=False)
