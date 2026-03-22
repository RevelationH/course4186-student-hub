from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional

from firebase_admin import firestore

from db import fire_db


SESSION_COLLECTION = "course4186_chat_sessions"
MESSAGE_COLLECTION = "messages"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        try:
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).isoformat()
        except Exception:
            return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return str(value)


TITLE_SANITIZE_RE = re.compile(r"[\r\n\t]+")


def _clean_title(text: str, fallback: str = "New chat") -> str:
    compact = TITLE_SANITIZE_RE.sub(" ", str(text or "").strip())
    compact = " ".join(compact.split()).strip(" -|:;,.\"'")
    if not compact:
        return fallback
    compact = compact[:56].strip()
    return compact or fallback


def _preview_text(text: str, limit: int = 120) -> str:
    compact = " ".join(str(text or "").strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


class ChatSessionStore:
    def __init__(self) -> None:
        self._fdb = fire_db()

    def _sessions_ref(self, user_id: str):
        return self._fdb.collection("users").document(user_id).collection(SESSION_COLLECTION)

    def _session_ref(self, user_id: str, session_id: str):
        return self._sessions_ref(user_id).document(session_id)

    def list_sessions(self, user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        docs = (
            self._sessions_ref(user_id)
            .order_by("updated_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        rows: List[Dict[str, Any]] = []
        for doc in docs:
            payload = doc.to_dict() or {}
            rows.append(self._serialize_session(doc.id, payload))
        return rows

    def get_session(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        doc = self._session_ref(user_id, session_id).get()
        if not doc.exists:
            return None
        return self._serialize_session(doc.id, doc.to_dict() or {})

    def create_session(self, user_id: str, title: str = "New chat") -> Dict[str, Any]:
        now = _now()
        ref = self._sessions_ref(user_id).document()
        payload = {
            "title": _clean_title(title),
            "title_generated": False,
            "created_at": now,
            "updated_at": now,
            "last_message_preview": "",
            "message_count": 0,
            "course": "4186",
        }
        ref.set(payload)
        return self._serialize_session(ref.id, payload)

    def get_or_create_session(self, user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        if session_id:
            existing = self.get_session(user_id, session_id)
            if existing:
                return existing
        return self.create_session(user_id)

    def get_messages(self, user_id: str, session_id: str) -> List[Dict[str, Any]]:
        session_ref = self._session_ref(user_id, session_id)
        if not session_ref.get().exists:
            return []
        docs = session_ref.collection(MESSAGE_COLLECTION).order_by("order_index").stream()
        rows: List[Dict[str, Any]] = []
        for doc in docs:
            payload = doc.to_dict() or {}
            rows.append(
                {
                    "message_id": doc.id,
                    "role": str(payload.get("role") or "assistant"),
                    "content": str(payload.get("content") or ""),
                    "citations": list(payload.get("citations") or []),
                    "created_at": _iso(payload.get("created_at")),
                    "order_index": int(payload.get("order_index") or 0),
                }
            )
        return rows

    def recent_history_for_model(self, user_id: str, session_id: str, limit: int = 8) -> List[Dict[str, str]]:
        rows = self.get_messages(user_id, session_id)
        trimmed = rows[-limit:]
        return [{"role": row["role"], "content": row["content"]} for row in trimmed if row.get("content")]

    def append_exchange(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        citations: List[Dict[str, Any]],
        mode: str,
        session_title: Optional[str] = None,
    ) -> Dict[str, Any]:
        session_ref = self._session_ref(user_id, session_id)
        session_doc = session_ref.get()
        if session_doc.exists:
            session_payload = session_doc.to_dict() or {}
        else:
            session_payload = {
                "title": "New chat",
                "title_generated": False,
                "created_at": _now(),
                "updated_at": _now(),
                "last_message_preview": "",
                "message_count": 0,
                "course": "4186",
            }
            session_ref.set(session_payload)

        message_count = int(session_payload.get("message_count") or 0)
        now = _now()
        base_title = _clean_title(str(session_payload.get("title") or ""), fallback="New chat")
        title_generated = bool(session_payload.get("title_generated"))
        if session_title and (message_count == 0 or base_title == "New chat" or not title_generated):
            base_title = _clean_title(session_title, fallback="New chat")
            title_generated = True

        user_payload = {
            "role": "user",
            "content": user_message,
            "citations": [],
            "created_at": now,
            "order_index": message_count + 1,
        }
        assistant_payload = {
            "role": "assistant",
            "content": assistant_message,
            "citations": list(citations or []),
            "created_at": now,
            "order_index": message_count + 2,
            "mode": mode,
        }
        session_ref.collection(MESSAGE_COLLECTION).document(f"m{message_count + 1:06d}").set(user_payload)
        session_ref.collection(MESSAGE_COLLECTION).document(f"m{message_count + 2:06d}").set(assistant_payload)

        session_payload.update(
            {
                "title": base_title,
                "title_generated": title_generated,
                "updated_at": now,
                "last_message_preview": _preview_text(assistant_message or user_message),
                "message_count": message_count + 2,
            }
        )
        session_ref.set(session_payload, merge=True)
        return self._serialize_session(session_id, session_payload)

    def set_session_title(
        self,
        user_id: str,
        session_id: str,
        title: str,
        *,
        generated: bool = True,
    ) -> Optional[Dict[str, Any]]:
        session_ref = self._session_ref(user_id, session_id)
        session_doc = session_ref.get()
        if not session_doc.exists:
            return None
        payload = session_doc.to_dict() or {}
        payload["title"] = _clean_title(title, fallback="New chat")
        payload["title_generated"] = bool(generated)
        payload["updated_at"] = _now()
        session_ref.set(payload, merge=True)
        return self._serialize_session(session_id, payload)

    def delete_session(self, user_id: str, session_id: str) -> bool:
        session_ref = self._session_ref(user_id, session_id)
        session_doc = session_ref.get()
        if not session_doc.exists:
            return False
        for doc in session_ref.collection(MESSAGE_COLLECTION).stream():
            doc.reference.delete()
        session_ref.delete()
        return True

    def _serialize_session(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "session_id": session_id,
            "title": str(payload.get("title") or "New chat"),
            "title_generated": bool(payload.get("title_generated")),
            "created_at": _iso(payload.get("created_at")),
            "updated_at": _iso(payload.get("updated_at")),
            "last_message_preview": str(payload.get("last_message_preview") or ""),
            "message_count": int(payload.get("message_count") or 0),
            "course": str(payload.get("course") or "4186"),
        }
