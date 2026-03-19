from __future__ import annotations

import os


KIMI_API_KEY = os.getenv("KIMI_API_KEY")
KIMI_API_BASE = os.getenv("KIMI_API_BASE", "https://api.moonshot.cn/v1")
KIMI_MODEL = os.getenv("KIMI_MODEL", "moonshot-v1-8k")
