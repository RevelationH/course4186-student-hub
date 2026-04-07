from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil


PORTAL_DIR = Path(__file__).resolve().parent
ROOT_DIR = PORTAL_DIR.parent
LOG_DIR = PORTAL_DIR / "logs"
STATUS_PATH = PORTAL_DIR / "status.json"
PORTAL_LOG = LOG_DIR / "portal.log"
TUNNEL_LOG = LOG_DIR / "tunnel.log"
ENV_PYTHON = Path(r"C:\Users\langhuang6\AppData\Local\anaconda3\envs\env_meta\python.exe")
CLOUDFLARED_EXE = ROOT_DIR / "public_launch" / "cloudflared.exe"
PORTAL_PORT = 50186
LOCAL_URL = f"http://127.0.0.1:{PORTAL_PORT}/healthz"
PUBLIC_URL_PATTERN = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com", re.IGNORECASE)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env_loader import load_project_env

load_project_env()

try:
    from kimi_utils import KIMI_API_BASE, KIMI_API_KEY, KIMI_MODEL
except Exception:
    KIMI_API_BASE = None
    KIMI_API_KEY = None
    KIMI_MODEL = None


def ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def http_ready(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


def process_is_running(pid: int) -> bool:
    try:
        proc = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return False
    return str(pid) in proc.stdout


def find_pid_listening_on_port(port: int) -> int | None:
    try:
        for conn in psutil.net_connections(kind="inet"):
            if not conn.laddr or conn.status != psutil.CONN_LISTEN:
                continue
            if conn.laddr.port == port and conn.pid:
                return int(conn.pid)
    except Exception:
        return None
    return None


def load_status() -> dict:
    if not STATUS_PATH.exists():
        return {}
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_status(data: dict) -> None:
    STATUS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def wait_for_local_service(timeout: int = 90) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_port_open("127.0.0.1", PORTAL_PORT) and http_ready(LOCAL_URL):
            return True
        time.sleep(1)
    return False


def start_portal_process() -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    preferred_artifact_dir = ROOT_DIR / "course4186_rag" / "artifacts_full_course"
    if preferred_artifact_dir.exists() and not env.get("COURSE4186_ARTIFACT_DIR"):
        env["COURSE4186_ARTIFACT_DIR"] = str(preferred_artifact_dir)
    if not any(env.get(name) for name in ("COURSE_LLM_API_KEY", "DEEPSEEK_API_KEY", "KIMI_API_KEY")) and KIMI_API_KEY:
        env.setdefault("COURSE_LLM_API_KEY", KIMI_API_KEY)
        if KIMI_API_BASE:
            env.setdefault("COURSE_LLM_BASE_URL", KIMI_API_BASE)
        if KIMI_MODEL:
            env.setdefault("COURSE_LLM_MODEL", KIMI_MODEL)
    portal_log = PORTAL_LOG.open("a", encoding="utf-8")
    return subprocess.Popen(
        [str(ENV_PYTHON), str(PORTAL_DIR / "run_4186_portal.py")],
        cwd=str(ROOT_DIR),
        env=env,
        stdout=portal_log,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )


def start_tunnel_process() -> subprocess.Popen:
    tunnel_log = TUNNEL_LOG.open("a", encoding="utf-8")
    return subprocess.Popen(
        [str(CLOUDFLARED_EXE), "tunnel", "--url", f"http://127.0.0.1:{PORTAL_PORT}", "--no-autoupdate"],
        cwd=str(PORTAL_DIR),
        stdout=tunnel_log,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )


def wait_for_public_url(timeout: int = 90) -> str | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if TUNNEL_LOG.exists():
            text = TUNNEL_LOG.read_text(encoding="utf-8", errors="ignore")
            match = PUBLIC_URL_PATTERN.search(text)
            if match:
                return match.group(0)
        time.sleep(1)
    return None


def main() -> int:
    ensure_dirs()
    status = load_status()

    portal_pid = status.get("portal_pid")
    portal_running = bool(portal_pid) and process_is_running(int(portal_pid)) and http_ready(LOCAL_URL)
    if not portal_running and is_port_open("127.0.0.1", PORTAL_PORT) and http_ready(LOCAL_URL):
        detected_pid = find_pid_listening_on_port(PORTAL_PORT)
        if detected_pid:
            portal_pid = detected_pid
            portal_running = True
    if not portal_running:
        print("Starting Course 4186 portal...", flush=True)
        portal_proc = start_portal_process()
        if not wait_for_local_service():
            print("ERROR: portal failed to become ready.", flush=True)
            print(f"Check log: {PORTAL_LOG}", flush=True)
            save_status(
                {
                    "portal_pid": portal_proc.pid,
                    "tunnel_pid": None,
                    "public_url": None,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "local_url": LOCAL_URL,
                    "portal_log": str(PORTAL_LOG),
                    "tunnel_log": str(TUNNEL_LOG),
                }
            )
            return 1
        portal_pid = portal_proc.pid
    else:
        print(f"Course 4186 portal already running on PID {portal_pid}.", flush=True)

    tunnel_pid = status.get("tunnel_pid")
    public_url = status.get("public_url")
    tunnel_running = bool(tunnel_pid) and process_is_running(int(tunnel_pid))
    if tunnel_running and public_url:
        save_status(
            {
                "portal_pid": int(portal_pid),
                "tunnel_pid": int(tunnel_pid),
                "public_url": public_url,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "local_url": LOCAL_URL,
                "portal_log": str(PORTAL_LOG),
                "tunnel_log": str(TUNNEL_LOG),
            }
        )
        print(public_url, flush=True)
        return 0

    print("Starting Cloudflare tunnel for Course 4186 portal...", flush=True)
    TUNNEL_LOG.write_text("", encoding="utf-8")
    tunnel_proc = start_tunnel_process()
    public_url = wait_for_public_url()

    save_status(
        {
            "portal_pid": int(portal_pid),
            "tunnel_pid": int(tunnel_proc.pid),
            "public_url": public_url,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "local_url": LOCAL_URL,
            "portal_log": str(PORTAL_LOG),
            "tunnel_log": str(TUNNEL_LOG),
        }
    )

    if not public_url:
        print("ERROR: tunnel started but no public URL was detected.", flush=True)
        print(f"Check log: {TUNNEL_LOG}", flush=True)
        return 1

    print(public_url, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
