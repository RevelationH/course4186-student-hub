from __future__ import annotations

import json
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class ProgressStore:
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        if not self.data_path.exists():
            self._save({"users": {}})

    def _load(self) -> Dict[str, Any]:
        with self.data_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save(self, payload: Dict[str, Any]) -> None:
        with self.data_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def ensure_user(
        self,
        user_id: str,
        display_name: Optional[str] = None,
        account_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            payload = self._load()
            users = payload.setdefault("users", {})
            default_name = f"Student-{user_id[:8]}"
            clean_display = self._clean_name(display_name)
            clean_account = self._clean_name(account_name)
            if user_id not in users:
                users[user_id] = {
                    "display_name": clean_display or clean_account or default_name,
                    "account_name": clean_account or clean_display or default_name,
                    "created_at": self._now_iso(),
                    "attempts": [],
                }
                self._save(payload)
            else:
                user = users[user_id]
                changed = False
                if not self._clean_name(user.get("account_name")):
                    user["account_name"] = self._clean_name(user.get("display_name")) or default_name
                    changed = True
                if clean_account:
                    previous_account = self._clean_name(user.get("account_name"))
                    user["account_name"] = clean_account
                    current_display = self._clean_name(user.get("display_name"))
                    if (not current_display) or current_display.startswith(("Learner-", "Student-")) or current_display == previous_account:
                        user["display_name"] = clean_account
                    changed = True
                elif clean_display:
                    user["display_name"] = clean_display
                    changed = True
                if changed:
                    self._save(payload)
            return dict(users[user_id])

    def set_display_name(self, user_id: str, display_name: str) -> Dict[str, Any]:
        clean_name = self._clean_name(display_name) or f"Student-{user_id[:8]}"
        return self.ensure_user(user_id, clean_name)

    def set_account_name(self, user_id: str, account_name: str) -> Dict[str, Any]:
        clean_name = self._clean_name(account_name) or f"Student-{user_id[:8]}"
        return self.ensure_user(user_id, account_name=clean_name)

    def get_user(self, user_id: str) -> Dict[str, Any]:
        return self.ensure_user(user_id)

    def all_attempts(self, user_id: str) -> List[Dict[str, Any]]:
        user = self.get_user(user_id)
        return [dict(item) for item in user.get("attempts", [])]

    def record_attempts(
        self,
        user_id: str,
        kp_id: str,
        kp_name: str,
        results: List[Dict[str, Any]],
    ) -> None:
        with self._lock:
            payload = self._load()
            users = payload.setdefault("users", {})
            user = users.setdefault(
                user_id,
                {
                    "display_name": f"Student-{user_id[:8]}",
                    "account_name": f"Student-{user_id[:8]}",
                    "created_at": self._now_iso(),
                    "attempts": [],
                },
            )
            attempts = user.setdefault("attempts", [])
            now = self._now_iso()
            for result in results:
                attempts.append(
                    {
                        "timestamp": now,
                        "kp_id": kp_id,
                        "kp_name": kp_name,
                        "question_id": result.get("question_id"),
                        "question_type": result.get("question_type"),
                        "question": result.get("question"),
                        "submitted_answer": result.get("submitted_answer"),
                        "reference_answer": result.get("reference_answer"),
                        "is_correct": bool(result.get("is_correct")),
                    }
                )
            self._save(payload)

    def summary(self, user_id: str) -> Dict[str, Any]:
        user = self.get_user(user_id)
        attempts = self._latest_attempts(user)
        answered = len(attempts)
        correct = sum(1 for item in attempts if item.get("is_correct"))
        wrong = answered - correct
        return {
            "answered": answered,
            "correct": correct,
            "wrong": wrong,
            "accuracy": round((correct / answered) * 100, 1) if answered else 0.0,
            "display_name": user.get("display_name"),
        }

    def kp_stats(self, user_id: str) -> List[Dict[str, Any]]:
        user = self.get_user(user_id)
        attempts = self._latest_attempts(user)
        grouped: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "kp_id": "",
                "kp_name": "",
                "answered": 0,
                "correct": 0,
                "wrong": 0,
                "last_at": "",
            }
        )
        for item in attempts:
            bucket = grouped[item["kp_id"]]
            bucket["kp_id"] = item["kp_id"]
            bucket["kp_name"] = item["kp_name"]
            bucket["answered"] += 1
            if item.get("is_correct"):
                bucket["correct"] += 1
            else:
                bucket["wrong"] += 1
            bucket["last_at"] = max(bucket["last_at"], item.get("timestamp") or "")

        rows = list(grouped.values())
        for row in rows:
            row["accuracy"] = round((row["correct"] / row["answered"]) * 100, 1) if row["answered"] else 0.0
        rows.sort(key=lambda row: (-row["wrong"], row["accuracy"], row["kp_name"]))
        return rows

    def weak_points(self, user_id: str, minimum_attempts: int = 1) -> List[Dict[str, Any]]:
        rows = [
            row for row in self.kp_stats(user_id)
            if row["answered"] >= minimum_attempts and (row["wrong"] > 0 or row["accuracy"] < 70)
        ]
        rows.sort(key=lambda row: (-row["wrong"], row["accuracy"], row["kp_name"]))
        return rows[:6]

    def recent_attempts(self, user_id: str, limit: int = 12) -> List[Dict[str, Any]]:
        user = self.get_user(user_id)
        attempts = list(user.get("attempts", []))
        attempts.sort(key=lambda row: row.get("timestamp") or "", reverse=True)
        return attempts[:limit]

    def _latest_attempts(self, user: Dict[str, Any]) -> List[Dict[str, Any]]:
        latest: Dict[tuple[str, str], Dict[str, Any]] = {}
        for item in user.get("attempts", []):
            key = (str(item.get("kp_id") or ""), str(item.get("question_id") or ""))
            previous = latest.get(key)
            if previous is None or (item.get("timestamp") or "") >= (previous.get("timestamp") or ""):
                latest[key] = item
        return list(latest.values())

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _clean_name(self, value: Optional[str]) -> str:
        return (value or "").strip()[:40]
