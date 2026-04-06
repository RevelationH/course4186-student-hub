from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from typing import Any, Dict, List, Optional

from firebase_admin import firestore

from db import fire_db
from course4186_portal.answer_consistency import (
    citation_course_source_line,
    extract_used_source_ids,
    normalize_answer_body_sources,
    rebuild_answer_with_citations,
    strip_course_source_line,
    strip_source_id_list_suffix,
)


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
COURSE_SOURCE_TOKEN_RE = re.compile(r"@@COURSE_SOURCE_\d+@@")
USED_SOURCES_RE = re.compile(r"(?is)(?:^|\n)\s*[\"']?used_sources[\"']?\s*:\s*\[[^\]]*\].*$")
TRANSPORT_WRAPPER_RE = re.compile(r"(?is)(^\s*```json\b|^\s*\{\s*[\"']?answer[\"']?\s*:|[\"']?used_sources[\"']?\s*:|@@COURSE_SOURCE_\d+@@)")
TRANSPORT_JSON_CODEBLOCK_RE = re.compile(r"(?is)\n```json\b[\s\S]*$")
TRANSPORT_JSON_SUFFIX_RE = re.compile(r"(?is)(?:\n|^)\s*\{\s*[\"']?answer[\"']?\s*:[\s\S]*$")


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


def _parse_jsonish(text: str) -> Any:
    candidate = str(text or "").strip()
    if not candidate:
        raise ValueError("Empty JSON response.")

    variants: List[str] = [candidate]
    if candidate.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
        if stripped:
            variants.append(stripped)

    decoder = json.JSONDecoder()
    seen: set[str] = set()
    for variant in variants:
        if not variant or variant in seen:
            continue
        seen.add(variant)
        for start_index in [0] + [match.start() for match in re.finditer(r"[\{\[]", variant)]:
            snippet = variant[start_index:].strip()
            if not snippet or snippet in seen:
                continue
            seen.add(snippet)
            try:
                parsed, _ = decoder.raw_decode(snippet)
                return parsed
            except Exception:
                continue
    raise ValueError("No JSON object found.")


def _citation_course_source_line(citations: List[Dict[str, Any]]) -> str:
    return citation_course_source_line(citations)


def _sanitize_assistant_content(text: str, citations: Optional[List[Dict[str, Any]]] = None) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""

    candidate = raw
    had_placeholder = bool(COURSE_SOURCE_TOKEN_RE.search(candidate))
    had_transport_wrapper = bool(
        TRANSPORT_WRAPPER_RE.search(candidate)
        or TRANSPORT_JSON_CODEBLOCK_RE.search(candidate)
        or TRANSPORT_JSON_SUFFIX_RE.search(candidate)
    )
    if TRANSPORT_WRAPPER_RE.search(candidate):
        try:
            parsed = _parse_jsonish(candidate)
            if isinstance(parsed, dict) and str(parsed.get("answer") or "").strip():
                candidate = str(parsed.get("answer") or "").strip()
        except Exception:
            pass

    json_answer_match = re.search(r'(?is)"answer"\s*:\s*"((?:\\.|[^"])*)"', candidate)
    if json_answer_match:
        try:
            candidate = json.loads(f"\"{json_answer_match.group(1)}\"")
        except Exception:
            candidate = json_answer_match.group(1)

    candidate = re.sub(r"^```json\s*", "", str(candidate).strip(), flags=re.IGNORECASE)
    candidate = re.sub(r"\s*```$", "", candidate).strip()
    candidate = USED_SOURCES_RE.sub("", candidate).strip()
    candidate = re.sub(r"(?is),?\s*[\"']?used_sources[\"']?\s*:\s*\[[^\]]*\]\s*\}?$", "", candidate).strip()
    candidate = COURSE_SOURCE_TOKEN_RE.sub("", candidate)
    candidate = re.sub(r"(?is)^\s*\{\s*[\"']?answer[\"']?\s*:\s*", "", candidate).strip()
    candidate = re.sub(r"(?is)\}\s*$", "", candidate).strip()

    cut_points: List[int] = []
    for pattern in (TRANSPORT_JSON_CODEBLOCK_RE, TRANSPORT_JSON_SUFFIX_RE):
        match = pattern.search(candidate)
        if match:
            prefix = candidate[: match.start()].strip()
            if prefix and re.search(r"[A-Za-z0-9\u4e00-\u9fff]", prefix):
                cut_points.append(match.start())
    if cut_points:
        candidate = candidate[: min(cut_points)].rstrip()

    candidate = strip_source_id_list_suffix(candidate)
    candidate = candidate.strip().strip(",").strip().strip("\"'")
    candidate = re.sub(r"\n{3,}", "\n\n", candidate).strip()

    if extract_used_source_ids(raw):
        had_transport_wrapper = True

    candidate = normalize_answer_body_sources(strip_course_source_line(candidate))

    if citations:
        candidate = rebuild_answer_with_citations(candidate, citations)
    elif had_placeholder or had_transport_wrapper:
        candidate = strip_course_source_line(candidate)

    return candidate or raw


def _sanitize_preview(text: str) -> str:
    raw = str(text or "")
    if TRANSPORT_WRAPPER_RE.search(raw):
        return _preview_text(_sanitize_assistant_content(raw))
    return raw


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
            sanitized_preview = _sanitize_preview(str(payload.get("last_message_preview") or ""))
            if sanitized_preview != str(payload.get("last_message_preview") or ""):
                payload["last_message_preview"] = sanitized_preview
                doc.reference.set({"last_message_preview": sanitized_preview}, merge=True)
            rows.append(self._serialize_session(doc.id, payload))
        return rows

    def get_session(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        doc = self._session_ref(user_id, session_id).get()
        if not doc.exists:
            return None
        payload = doc.to_dict() or {}
        sanitized_preview = _sanitize_preview(str(payload.get("last_message_preview") or ""))
        if sanitized_preview != str(payload.get("last_message_preview") or ""):
            payload["last_message_preview"] = sanitized_preview
            doc.reference.set({"last_message_preview": sanitized_preview}, merge=True)
        return self._serialize_session(doc.id, payload)

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
        session_doc = session_ref.get()
        if not session_doc.exists:
            return []
        docs = session_ref.collection(MESSAGE_COLLECTION).order_by("order_index").stream()
        rows: List[Dict[str, Any]] = []
        for doc in docs:
            payload = doc.to_dict() or {}
            role = str(payload.get("role") or "assistant")
            citations = list(payload.get("citations") or [])
            content = str(payload.get("content") or "")
            if role == "assistant":
                sanitized_content = _sanitize_assistant_content(content, citations)
                if sanitized_content != content:
                    content = sanitized_content
                    doc.reference.set({"content": sanitized_content}, merge=True)
            rows.append(
                {
                    "message_id": doc.id,
                    "role": role,
                    "content": content,
                    "citations": citations,
                    "created_at": _iso(payload.get("created_at")),
                    "order_index": int(payload.get("order_index") or 0),
                }
            )
        repaired_preview = _preview_text(next((row["content"] for row in reversed(rows) if row.get("content")), ""))
        stored_preview = str((session_doc.to_dict() or {}).get("last_message_preview") or "")
        if repaired_preview and repaired_preview != stored_preview:
            session_ref.set({"last_message_preview": repaired_preview}, merge=True)
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

        assistant_message = _sanitize_assistant_content(assistant_message, list(citations or []))

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

    def repair_all_sessions(self) -> Dict[str, int]:
        scanned_users = 0
        scanned_sessions = 0
        repaired_messages = 0
        repaired_previews = 0

        for user_doc in self._fdb.collection("users").stream():
            scanned_users += 1
            user_id = user_doc.id
            for session_doc in self._sessions_ref(user_id).stream():
                scanned_sessions += 1
                session_ref = session_doc.reference
                session_payload = session_doc.to_dict() or {}
                last_content = ""
                for message_doc in session_ref.collection(MESSAGE_COLLECTION).order_by("order_index").stream():
                    payload = message_doc.to_dict() or {}
                    role = str(payload.get("role") or "assistant")
                    if role != "assistant":
                        if str(payload.get("content") or "").strip():
                            last_content = str(payload.get("content") or "")
                        continue
                    citations = list(payload.get("citations") or [])
                    content = str(payload.get("content") or "")
                    sanitized_content = _sanitize_assistant_content(content, citations)
                    if sanitized_content != content:
                        message_doc.reference.set({"content": sanitized_content}, merge=True)
                        repaired_messages += 1
                    if sanitized_content.strip():
                        last_content = sanitized_content
                repaired_preview = _preview_text(last_content or str(session_payload.get("last_message_preview") or ""))
                stored_preview = str(session_payload.get("last_message_preview") or "")
                if repaired_preview and repaired_preview != stored_preview:
                    session_ref.set({"last_message_preview": repaired_preview}, merge=True)
                    repaired_previews += 1

        return {
            "users": scanned_users,
            "sessions": scanned_sessions,
            "repaired_messages": repaired_messages,
            "repaired_previews": repaired_previews,
        }

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
