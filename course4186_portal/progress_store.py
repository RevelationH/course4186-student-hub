from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from db import fire_db


PROGRESS_USER_COLLECTION = "course4186_progress_users"
ATTEMPT_COLLECTION = "attempts"
ARTIFACT_COLLECTION = "artifacts"
FOLLOW_UP_DOC_ID = "generated_followups"
MIGRATION_COLLECTION = "course4186_portal_meta"
MIGRATION_DOC_ID = "progress_json_to_firestore_v1"
DEFAULT_BATCH_LIMIT = 425


class ProgressStore:
    def __init__(self, data_path: Optional[Path] = None):
        self.legacy_data_path = Path(data_path) if data_path else None
        self._fdb = fire_db()
        self._db = self._fdb.db
        self._migrate_legacy_json_if_needed()

    def ensure_user(
        self,
        user_id: str,
        display_name: Optional[str] = None,
        account_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        user_id = self._clean_user_id(user_id)
        if not user_id:
            raise ValueError("Missing user_id.")

        ref = self._user_ref(user_id)
        doc = ref.get()
        default_name = f"Student-{user_id[:8]}"
        clean_display = self._clean_name(display_name)
        clean_account = self._clean_name(account_name)

        if doc.exists:
            payload = doc.to_dict() or {}
        else:
            payload = {
                "display_name": clean_display or clean_account or default_name,
                "account_name": clean_account or clean_display or default_name,
                "created_at": self._now_iso(),
                "updated_at": self._now_iso(),
            }
            ref.set(payload, merge=True)
            return self._serialize_user(payload, user_id)

        changed = False
        if not self._clean_name(payload.get("account_name")):
            payload["account_name"] = self._clean_name(payload.get("display_name")) or default_name
            changed = True

        if not self._clean_name(payload.get("display_name")):
            payload["display_name"] = self._clean_name(payload.get("account_name")) or default_name
            changed = True

        if not str(payload.get("created_at") or "").strip():
            payload["created_at"] = self._now_iso()
            changed = True

        if clean_account:
            previous_account = self._clean_name(payload.get("account_name"))
            payload["account_name"] = clean_account
            current_display = self._clean_name(payload.get("display_name"))
            if (not current_display) or current_display.startswith(("Learner-", "Student-")) or current_display == previous_account:
                payload["display_name"] = clean_account
            changed = True
        elif clean_display:
            payload["display_name"] = clean_display
            changed = True

        if changed:
            payload["updated_at"] = self._now_iso()
            ref.set(payload, merge=True)

        return self._serialize_user(payload, user_id)

    def set_display_name(self, user_id: str, display_name: str) -> Dict[str, Any]:
        clean_name = self._clean_name(display_name) or f"Student-{self._clean_user_id(user_id)[:8]}"
        return self.ensure_user(user_id, clean_name)

    def set_account_name(self, user_id: str, account_name: str) -> Dict[str, Any]:
        clean_name = self._clean_name(account_name) or f"Student-{self._clean_user_id(user_id)[:8]}"
        return self.ensure_user(user_id, account_name=clean_name)

    def get_user(self, user_id: str) -> Dict[str, Any]:
        return self.ensure_user(user_id)

    def all_attempts(self, user_id: str) -> List[Dict[str, Any]]:
        user_id = self._clean_user_id(user_id)
        docs = self._attempts_ref(user_id).order_by("timestamp").stream()
        rows: List[Dict[str, Any]] = []
        for doc in docs:
            payload = doc.to_dict() or {}
            row = self._serialize_attempt(payload)
            row["attempt_id"] = doc.id
            rows.append(row)
        return rows

    def record_attempts(
        self,
        user_id: str,
        kp_id: str,
        kp_name: str,
        results: List[Dict[str, Any]],
    ) -> None:
        user_id = self._clean_user_id(user_id)
        if not results:
            return

        profile = self.ensure_user(user_id)
        now = self._now_iso()
        batch = self._db.batch()
        operation_count = 0

        for result in results:
            payload = {
                "timestamp": now,
                "kp_id": str(kp_id or ""),
                "kp_name": str(kp_name or ""),
                "question_id": str(result.get("question_id") or ""),
                "question_type": str(result.get("question_type") or ""),
                "question": str(result.get("question") or ""),
                "submitted_answer": str(result.get("submitted_answer") or ""),
                "reference_answer": str(result.get("reference_answer") or ""),
                "is_correct": bool(result.get("is_correct")),
            }
            batch.set(self._attempts_ref(user_id).document(), payload)
            operation_count += 1
            if operation_count >= DEFAULT_BATCH_LIMIT:
                batch.commit()
                batch = self._db.batch()
                operation_count = 0

        profile_payload = {
            "display_name": profile.get("display_name") or f"Student-{user_id[:8]}",
            "account_name": profile.get("account_name") or profile.get("display_name") or f"Student-{user_id[:8]}",
            "created_at": profile.get("created_at") or now,
            "updated_at": now,
            "last_attempt_at": now,
        }
        batch.set(self._user_ref(user_id), profile_payload, merge=True)
        batch.set(
            self._follow_up_ref(user_id),
            {
                "cache_key": "",
                "created_at": now,
                "items": [],
            },
        )
        batch.commit()

    def get_generated_followups(self, user_id: str, cache_key: str) -> List[Dict[str, Any]]:
        if not str(cache_key or "").strip():
            return []
        user_id = self._clean_user_id(user_id)
        payload = self._follow_up_ref(user_id).get()
        if not payload.exists:
            return []
        doc = payload.to_dict() or {}
        if str(doc.get("cache_key") or "") != str(cache_key or ""):
            return []
        return [self._sanitize_nested_item(item) for item in doc.get("items", []) if isinstance(item, dict)]

    def set_generated_followups(self, user_id: str, cache_key: str, items: List[Dict[str, Any]]) -> None:
        user_id = self._clean_user_id(user_id)
        profile = self.ensure_user(user_id)
        now = self._now_iso()
        self._follow_up_ref(user_id).set(
            {
                "cache_key": str(cache_key or ""),
                "created_at": now,
                "items": [self._sanitize_nested_item(item) for item in items if isinstance(item, dict)],
            }
        )
        self._user_ref(user_id).set(
            {
                "display_name": profile.get("display_name") or f"Student-{user_id[:8]}",
                "account_name": profile.get("account_name") or profile.get("display_name") or f"Student-{user_id[:8]}",
                "created_at": profile.get("created_at") or now,
                "updated_at": now,
            },
            merge=True,
        )

    def summary(self, user_id: str) -> Dict[str, Any]:
        user = self.get_user(user_id)
        attempts = self._latest_attempts(self.all_attempts(user_id))
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
        attempts = self._latest_attempts(self.all_attempts(user_id))
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
        attempts = self.all_attempts(user_id)
        attempts.sort(key=lambda row: row.get("timestamp") or "", reverse=True)
        return attempts[:limit]

    def _user_ref(self, user_id: str):
        return self._fdb.collection(PROGRESS_USER_COLLECTION).document(user_id)

    def _attempts_ref(self, user_id: str):
        return self._user_ref(user_id).collection(ATTEMPT_COLLECTION)

    def _follow_up_ref(self, user_id: str):
        return self._user_ref(user_id).collection(ARTIFACT_COLLECTION).document(FOLLOW_UP_DOC_ID)

    def _migration_ref(self):
        return self._fdb.collection(MIGRATION_COLLECTION).document(MIGRATION_DOC_ID)

    def _migrate_legacy_json_if_needed(self) -> None:
        if not self.legacy_data_path or not self.legacy_data_path.exists():
            return

        try:
            raw_payload = json.loads(self.legacy_data_path.read_text(encoding="utf-8"))
        except Exception:
            return

        users = raw_payload.get("users")
        if not isinstance(users, dict) or not users:
            return

        stat = self.legacy_data_path.stat()
        fingerprint = self._migration_fingerprint(self.legacy_data_path, stat.st_size, stat.st_mtime_ns)
        migration_doc = self._migration_ref().get()
        if migration_doc.exists:
            existing = migration_doc.to_dict() or {}
            if str(existing.get("source_fingerprint") or "") == fingerprint:
                return

        batch = self._db.batch()
        operation_count = 0
        user_count = 0
        attempt_count = 0
        followup_count = 0

        def queue_set(ref: Any, payload: Dict[str, Any], *, merge: bool = False) -> None:
            nonlocal batch, operation_count
            batch.set(ref, payload, merge=merge)
            operation_count += 1
            if operation_count >= DEFAULT_BATCH_LIMIT:
                batch.commit()
                batch = self._db.batch()
                operation_count = 0

        for user_id, payload in users.items():
            clean_user_id = self._clean_user_id(user_id)
            if not clean_user_id or not isinstance(payload, dict):
                continue

            user_count += 1
            default_name = f"Student-{clean_user_id[:8]}"
            user_doc = {
                "display_name": self._clean_name(payload.get("display_name")) or self._clean_name(payload.get("account_name")) or default_name,
                "account_name": self._clean_name(payload.get("account_name")) or self._clean_name(payload.get("display_name")) or default_name,
                "created_at": str(payload.get("created_at") or self._now_iso()),
                "updated_at": self._now_iso(),
                "migrated_from_legacy_json": True,
            }
            queue_set(self._user_ref(clean_user_id), user_doc, merge=True)

            attempts = payload.get("attempts") or []
            for index, attempt in enumerate(attempts):
                if not isinstance(attempt, dict):
                    continue
                attempt_doc = self._serialize_attempt(attempt)
                if not attempt_doc.get("timestamp"):
                    attempt_doc["timestamp"] = self._now_iso()
                doc_id = self._legacy_attempt_id(clean_user_id, index, attempt_doc)
                queue_set(self._attempts_ref(clean_user_id).document(doc_id), attempt_doc, merge=True)
                attempt_count += 1

            followups = payload.get("generated_followups")
            if isinstance(followups, dict):
                followup_doc = {
                    "cache_key": str(followups.get("cache_key") or ""),
                    "created_at": str(followups.get("created_at") or ""),
                    "items": [self._sanitize_nested_item(item) for item in followups.get("items", []) if isinstance(item, dict)],
                }
                queue_set(self._follow_up_ref(clean_user_id), followup_doc, merge=False)
                followup_count += len(followup_doc["items"])

        if operation_count:
            batch.commit()

        self._migration_ref().set(
            {
                "source_path": str(self.legacy_data_path),
                "source_fingerprint": fingerprint,
                "migrated_at": self._now_iso(),
                "user_count": user_count,
                "attempt_count": attempt_count,
                "followup_count": followup_count,
            },
            merge=True,
        )

    def _migration_fingerprint(self, path: Path, size: int, mtime_ns: int) -> str:
        raw = f"{path.resolve()}|{size}|{mtime_ns}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _legacy_attempt_id(self, user_id: str, index: int, attempt: Dict[str, Any]) -> str:
        raw = "|".join(
            [
                user_id,
                str(index),
                str(attempt.get("timestamp") or ""),
                str(attempt.get("kp_id") or ""),
                str(attempt.get("question_id") or ""),
                str(attempt.get("submitted_answer") or ""),
                str(attempt.get("reference_answer") or ""),
            ]
        )
        return f"legacy-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:24]}"

    def _latest_attempts(self, attempts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        latest: Dict[tuple[str, str], Dict[str, Any]] = {}
        for item in attempts:
            key = (str(item.get("kp_id") or ""), str(item.get("question_id") or ""))
            previous = latest.get(key)
            if previous is None or (item.get("timestamp") or "") >= (previous.get("timestamp") or ""):
                latest[key] = item
        return list(latest.values())

    def _serialize_user(self, payload: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        default_name = f"Student-{user_id[:8]}"
        display_name = self._clean_name(payload.get("display_name")) or self._clean_name(payload.get("account_name")) or default_name
        account_name = self._clean_name(payload.get("account_name")) or display_name
        return {
            "display_name": display_name,
            "account_name": account_name,
            "created_at": str(payload.get("created_at") or ""),
            "updated_at": str(payload.get("updated_at") or ""),
        }

    def _serialize_attempt(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "timestamp": str(payload.get("timestamp") or ""),
            "kp_id": str(payload.get("kp_id") or ""),
            "kp_name": str(payload.get("kp_name") or ""),
            "question_id": str(payload.get("question_id") or ""),
            "question_type": str(payload.get("question_type") or ""),
            "question": str(payload.get("question") or ""),
            "submitted_answer": str(payload.get("submitted_answer") or ""),
            "reference_answer": str(payload.get("reference_answer") or ""),
            "is_correct": bool(payload.get("is_correct")),
        }

    def _sanitize_nested_item(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._sanitize_nested_item(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._sanitize_nested_item(item) for item in value]
        if isinstance(value, tuple):
            return [self._sanitize_nested_item(item) for item in value]
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, datetime):
            return self._datetime_to_iso(value)
        return value

    def _datetime_to_iso(self, value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _clean_name(self, value: Optional[str]) -> str:
        return (value or "").strip()[:40]

    def _clean_user_id(self, value: Optional[str]) -> str:
        return (value or "").strip()
