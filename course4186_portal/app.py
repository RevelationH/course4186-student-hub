from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, session, url_for


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
COURSE_MATERIAL_ROOT = ROOT_DIR / "4186" / "4186"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from course4186_portal.kb_service import Course4186KnowledgeBase, STOPWORDS
from course4186_portal.chat_session_store import ChatSessionStore
from course4186_portal.progress_store import ProgressStore
from course4186_portal.student_analytics import build_dashboard_context, build_learning_report_context

try:
    from user import User
    from werkzeug.security import check_password_hash, generate_password_hash
    AUTH_BACKEND_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - defensive import fallback
    User = None
    check_password_hash = None
    generate_password_hash = None
    AUTH_BACKEND_ERROR = exc


ANSWER_LABEL_RE = re.compile(r"\b([A-D])\b", flags=re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*")
PUBLIC_ENDPOINTS = {
    "healthz",
    "login_4186",
    "register_4186",
    "logout_4186",
    "course4186_web_asset",
    "static",
}
RAW_CHAT_TITLE_PREFIXES = (
    "what is", "what are", "explain", "how", "why", "who", "when", "where",
    "can you", "tell me", "compare", "introduce", "briefly explain",
)
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{2,40}$")


def clean_account_name(value: Optional[str]) -> str:
    return (value or "").strip()[:40]


def clean_login_name(value: Optional[str]) -> str:
    return (value or "").strip()


def is_portal_authenticated() -> bool:
    return bool(session.get("course4186_logged_in") and clean_login_name(session.get("course4186_username")))


def safe_next_path(candidate: Optional[str]) -> str:
    value = (candidate or "").strip()
    if not value:
        return ""
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc:
        return ""
    if not value.startswith("/"):
        return ""
    return value


def login_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    return f"Login service is temporarily unavailable: {message}" if message else "Login service is temporarily unavailable. Please try again later."


def registration_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    return f"Registration service is temporarily unavailable: {message}" if message else "Registration service is temporarily unavailable. Please try again later."


def validate_registration_form(username: str, password: str, confirm_password: str) -> Optional[str]:
    if not username or not password or not confirm_password:
        return "Please complete all registration fields."
    if not USERNAME_RE.fullmatch(username):
        return "Use 2 to 40 characters for the account name. Letters, numbers, dot, underscore, and hyphen are allowed."
    if password != confirm_password:
        return "The two password entries do not match."
    if len(password) < 2:
        return "Please use a password with at least 2 characters."
    return None


def should_refresh_chat_title(
    title: Optional[str],
    *,
    title_generated: bool = False,
    message_count: int = 0,
) -> bool:
    text = " ".join(str(title or "").strip().split())
    lowered = text.lower()
    if not title_generated and int(message_count or 0) >= 2:
        return True
    if not text or text == "New chat":
        return True
    if len(text) > 42:
        return True
    if "?" in text or "？" in text:
        return True
    if lowered.startswith(RAW_CHAT_TITLE_PREFIXES):
        return True
    return False


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(APP_DIR / "templates"))
    app.config["SECRET_KEY"] = os.getenv("COURSE4186_PORTAL_SECRET", "course4186-portal-secret")
    app.config["SESSION_COOKIE_NAME"] = "course4186_portal_session"
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    kb = Course4186KnowledgeBase()
    store = ProgressStore(APP_DIR / "data" / "progress.json")
    chat_store = ChatSessionStore()

    app.config["kb"] = kb
    app.config["store"] = store
    app.config["chat_store"] = chat_store

    def render_login_page(
        *,
        error_message: str = "",
        success_message: str = "",
        username_value: str = "",
        next_path: str = "",
        form_mode: str = "login",
        status_code: int = 200,
    ) -> Any:
        resolved_next = safe_next_path(next_path) or url_for("chatapi_4186")
        return (
            render_template(
                "login_4186.html",
                error_message=error_message,
                success_message=success_message,
                username_value=username_value,
                next_path=resolved_next,
                auth_backend_ready=AUTH_BACKEND_ERROR is None,
                form_mode=form_mode if form_mode in {"login", "register"} else "login",
                login_url=url_for("login_4186"),
                register_url=url_for("register_4186"),
            ),
            status_code,
        )

    @app.before_request
    def ensure_course_user() -> Optional[Any]:
        endpoint = request.endpoint or ""
        if endpoint in PUBLIC_ENDPOINTS:
            return None

        if not is_portal_authenticated():
            if request.path.startswith("/api/4186/"):
                return jsonify(
                    {
                        "ok": False,
                        "error": "Authentication required.",
                        "login_url": url_for("login_4186"),
                    }
                ), 401

            next_path = request.full_path[:-1] if request.full_path.endswith("?") else request.full_path
            return redirect(url_for("login_4186", next=next_path))

        username = clean_login_name(session.get("course4186_username"))
        if not username:
            session.clear()
            return redirect(url_for("login_4186"))

        session["course4186_user_id"] = username
        session["course4186_account_name"] = clean_account_name(username) or username
        store.ensure_user(username, display_name=username, account_name=username)
        return None

    @app.context_processor
    def inject_course_nav() -> Dict[str, Any]:
        if not is_portal_authenticated():
            return {
                "course4186_profile": {"display_name": "Student", "account_name": ""},
                "course4186_summary": {"answered": 0, "accuracy": 0.0},
            }

        user_id = clean_login_name(session.get("course4186_user_id"))
        dashboard_context = build_dashboard_context(kb, store, user_id) if user_id else {"summary": {"answered": 0, "accuracy": 0.0}}
        summary = dashboard_context["summary"]
        profile = store.get_user(user_id) if user_id else {"display_name": "Student", "account_name": ""}
        return {"course4186_profile": profile, "course4186_summary": summary}

    @app.get("/")
    def index() -> Any:
        return redirect(url_for("chatapi_4186"))

    @app.route("/course4186/login", methods=["GET", "POST"])
    @app.route("/login_4186", methods=["GET", "POST"])
    def login_4186() -> Any:
        if is_portal_authenticated():
            redirect_target = safe_next_path(request.values.get("next")) or url_for("chatapi_4186")
            return redirect(redirect_target)

        if request.method == "GET":
            success_message = ""
            if request.args.get("registered") == "1":
                success_message = "Registration successful. Please sign in with your new account."
            return render_login_page(
                success_message=success_message,
                username_value=clean_login_name(request.args.get("username", "")),
                next_path=request.args.get("next", ""),
                form_mode="login",
            )

        username = clean_login_name(request.form.get("username"))
        password = request.form.get("password") or ""
        next_path = request.form.get("next") or ""

        if AUTH_BACKEND_ERROR is not None or User is None or check_password_hash is None:
            return render_login_page(
                error_message=login_error_message(AUTH_BACKEND_ERROR or RuntimeError("Unknown backend error")),
                username_value=username,
                next_path=next_path,
                form_mode="login",
                status_code=503,
            )

        if not username or not password:
            return render_login_page(
                error_message="Please enter both username and password.",
                username_value=username,
                next_path=next_path,
                form_mode="login",
                status_code=400,
            )

        try:
            user = User.get_by_username(username)
        except Exception as exc:
            return render_login_page(
                error_message=login_error_message(exc),
                username_value=username,
                next_path=next_path,
                form_mode="login",
                status_code=503,
            )

        if not user or not check_password_hash(user.password, password):
            return render_login_page(
                error_message="Incorrect username or password.",
                username_value=username,
                next_path=next_path,
                form_mode="login",
                status_code=401,
            )

        session.clear()
        session["course4186_logged_in"] = True
        session["course4186_user_id"] = user.username
        session["course4186_username"] = user.username
        session["course4186_is_admin"] = bool(getattr(user, "is_admin", False))
        session["course4186_account_name"] = clean_account_name(user.username) or user.username
        store.ensure_user(user.username, display_name=user.username, account_name=user.username)
        return redirect(safe_next_path(next_path) or url_for("chatapi_4186"))

    @app.route("/course4186/register", methods=["GET", "POST"])
    @app.route("/register_4186", methods=["GET", "POST"])
    def register_4186() -> Any:
        if is_portal_authenticated():
            redirect_target = safe_next_path(request.values.get("next")) or url_for("chatapi_4186")
            return redirect(redirect_target)

        if request.method == "GET":
            return render_login_page(
                username_value=clean_login_name(request.args.get("username", "")),
                next_path=request.args.get("next", ""),
                form_mode="register",
            )

        username = clean_login_name(request.form.get("username"))
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""
        next_path = request.form.get("next") or ""

        if AUTH_BACKEND_ERROR is not None or User is None or generate_password_hash is None:
            return render_login_page(
                error_message=registration_error_message(AUTH_BACKEND_ERROR or RuntimeError("Unknown backend error")),
                username_value=username,
                next_path=next_path,
                form_mode="register",
                status_code=503,
            )

        validation_error = validate_registration_form(username, password, confirm_password)
        if validation_error:
            return render_login_page(
                error_message=validation_error,
                username_value=username,
                next_path=next_path,
                form_mode="register",
                status_code=400,
            )

        try:
            existing_user = User.get_by_username(username)
        except Exception as exc:
            return render_login_page(
                error_message=registration_error_message(exc),
                username_value=username,
                next_path=next_path,
                form_mode="register",
                status_code=503,
            )

        if existing_user:
            return render_login_page(
                error_message="This account name is already in use. Please choose another one.",
                username_value=username,
                next_path=next_path,
                form_mode="register",
                status_code=409,
            )

        try:
            user = User(username, generate_password_hash(password), False)
            user.save()
        except Exception as exc:
            return render_login_page(
                error_message=registration_error_message(exc),
                username_value=username,
                next_path=next_path,
                form_mode="register",
                status_code=503,
            )

        login_redirect = url_for(
            "login_4186",
            registered="1",
            username=username,
            next=safe_next_path(next_path) or url_for("chatapi_4186"),
        )
        return redirect(login_redirect)

    @app.get("/course4186/logout")
    @app.get("/logout_4186")
    def logout_4186() -> Any:
        session.clear()
        return redirect(url_for("login_4186"))

    @app.get("/healthz")
    def healthz() -> Any:
        return jsonify(
            {
                "ok": True,
                "knowledge_points": len(kb.kps),
                "questions": len(kb.questions),
            }
        )

    @app.get("/chatapi_4186")
    @app.get("/chatapi_4186.html")
    def chatapi_4186() -> Any:
        profile = store.get_user(session["course4186_user_id"])
        return render_template(
            "chatapi_4186.html",
            display_name=profile["display_name"],
            dense_enabled=kb.dense_enabled,
        )

    @app.get("/chatapi_4186_classic")
    @app.get("/chatapi_4186_classic.html")
    def chatapi_4186_classic() -> Any:
        return render_template("chatapi_4186_classic.html")

    @app.get("/course4186/web/<path:filename>")
    def course4186_web_asset(filename: str) -> Any:
        return send_from_directory(ROOT_DIR / "web", filename)

    @app.get("/course4186/materials/<path:filename>")
    def course4186_material_asset(filename: str) -> Any:
        return send_from_directory(COURSE_MATERIAL_ROOT, filename)

    @app.get("/course4186/artifacts/<path:filename>")
    def course4186_artifact_asset(filename: str) -> Any:
        return send_from_directory(kb.artifact_dir, filename)

    @app.post("/api/4186/profile")
    def update_profile() -> Any:
        payload = request.get_json(silent=True) or {}
        profile = store.set_display_name(session["course4186_user_id"], payload.get("display_name", ""))
        return jsonify({"ok": True, "profile": profile, "summary": store.summary(session["course4186_user_id"])})

    @app.post("/api/4186/chat")
    def api_chat_4186() -> Any:
        payload = request.get_json(silent=True) or {}
        message = (payload.get("message") or "").strip()
        requested_history = payload.get("history") or []
        session_id = str(payload.get("session_id") or "").strip()
        if not message:
            return jsonify({"ok": False, "error": "Please enter a question before sending."}), 400

        user_id = session["course4186_user_id"]
        if session_id:
            history = chat_store.recent_history_for_model(user_id, session_id, limit=8)
        else:
            history = requested_history[-8:] if isinstance(requested_history, list) else []

        answer = kb.answer(message, history=history, top_k=5)
        citations = []
        for item in answer["citations"]:
            normalized_source = str(item.get("source") or "").replace("\\", "/")
            enriched = dict(item)
            enriched["material_url"] = url_for("course4186_material_asset", filename=normalized_source)
            citations.append(enriched)
        session_data = chat_store.get_or_create_session(user_id, session_id=session_id)
        generated_title = kb.suggest_session_title(message, answer["answer"])
        session_data = chat_store.append_exchange(
            user_id=user_id,
            session_id=session_data["session_id"],
            user_message=message,
            assistant_message=answer["answer"],
            citations=citations,
            mode=answer["mode"],
            session_title=generated_title,
        )
        return jsonify(
            {
                "ok": True,
                "answer": answer["answer"],
                "citations": citations,
                "related_kps": answer["related_kps"],
                "mode": answer["mode"],
                "actions": {
                    "quiz_url": url_for("quiz_dashboard_4186"),
                },
                "session": session_data,
            }
        )

    @app.get("/api/4186/chat/sessions")
    def api_chat_sessions_4186() -> Any:
        user_id = session["course4186_user_id"]
        sessions = chat_store.list_sessions(user_id)
        refreshed = False
        for item in sessions:
            if not should_refresh_chat_title(
                item.get("title"),
                title_generated=bool(item.get("title_generated")),
                message_count=int(item.get("message_count") or 0),
            ):
                continue
            messages = chat_store.get_messages(user_id, item["session_id"])
            user_message = next((row.get("content", "") for row in messages if row.get("role") == "user" and row.get("content")), "")
            assistant_message = next((row.get("content", "") for row in messages if row.get("role") == "assistant" and row.get("content")), "")
            if not user_message:
                continue
            suggested = kb.suggest_session_title(user_message, assistant_message)
            updated = chat_store.set_session_title(user_id, item["session_id"], suggested, generated=True)
            if updated:
                refreshed = True
        if refreshed:
            sessions = chat_store.list_sessions(user_id)
        return jsonify({"ok": True, "sessions": sessions})

    @app.post("/api/4186/chat/sessions")
    def api_create_chat_session_4186() -> Any:
        payload = request.get_json(silent=True) or {}
        session_data = chat_store.create_session(
            session["course4186_user_id"],
            title=str(payload.get("title") or "New chat"),
        )
        return jsonify({"ok": True, "session": session_data}), 201

    @app.get("/api/4186/chat/sessions/<string:session_id>")
    def api_chat_session_detail_4186(session_id: str) -> Any:
        session_data = chat_store.get_session(session["course4186_user_id"], session_id)
        if not session_data:
            return jsonify({"ok": False, "error": "Conversation not found."}), 404
        messages = chat_store.get_messages(session["course4186_user_id"], session_id)
        return jsonify({"ok": True, "session": session_data, "messages": messages})

    @app.delete("/api/4186/chat/sessions/<string:session_id>")
    def api_delete_chat_session_4186(session_id: str) -> Any:
        deleted = chat_store.delete_session(session["course4186_user_id"], session_id)
        if not deleted:
            return jsonify({"ok": False, "error": "Conversation not found."}), 404
        return jsonify({"ok": True})

    @app.get("/quiz_4186")
    def quiz_dashboard_4186() -> Any:
        dashboard_context = build_dashboard_context(kb, store, session["course4186_user_id"])
        return render_template(
            "quiz_dashboard_4186.html",
            kps=dashboard_context["kps"],
            profile=store.get_user(session["course4186_user_id"]),
        )

    @app.route("/quiz_4186/practice/<string:kp_id>", methods=["GET", "POST"])
    def quiz_practice_4186(kp_id: str) -> Any:
        kp = kb.get_kp(kp_id)
        if not kp:
            return "Knowledge point not found.", 404

        questions = kp.get("questions", [])
        results_map: Dict[str, Dict[str, Any]] = {}
        summary = None
        practice_note = "This practice set uses lecture-based multiple-choice questions. Submit your answers to see the explanation and the recommended slides or pages for review."

        if request.method == "POST":
            submission_rows: List[Dict[str, Any]] = []
            answered = 0
            correct = 0
            for question in questions:
                field_name = f"answer_{question['question_id']}"
                user_answer = (request.form.get(field_name) or "").strip()
                result = grade_question(question, kp, user_answer)
                results_map[question["question_id"]] = result
                if user_answer:
                    answered += 1
                    if result["is_correct"]:
                        correct += 1
                    submission_rows.append(
                        {
                            "question_id": question["question_id"],
                            "question_type": question.get("question_type"),
                            "question": question.get("question"),
                            "submitted_answer": user_answer,
                            "reference_answer": question.get("correct_option") or question.get("answer"),
                            "is_correct": result["is_correct"],
                        }
                    )

            summary = {
                "answered": answered,
                "correct": correct,
                "wrong": max(answered - correct, 0),
                "accuracy": round((correct / answered) * 100, 1) if answered else 0.0,
            }
            if submission_rows:
                store.record_attempts(session["course4186_user_id"], kp_id=kp["kp_id"], kp_name=kp["name"], results=submission_rows)

        return render_template(
            "quiz_practice_4186.html",
            kp=kp,
            questions=questions,
            results_map=results_map,
            summary=summary,
            practice_note=practice_note,
        )

    @app.get("/quiz_4186/analysis")
    def quiz_analysis_4186() -> Any:
        report_context = build_learning_report_context(kb, store, session["course4186_user_id"])
        return render_template(
            "quiz_analysis_4186.html",
            profile=store.get_user(session["course4186_user_id"]),
            **report_context,
        )

    return app


def normalize_text(text: str) -> str:
    compact = re.sub(r"[^a-z0-9\s]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", compact).strip()


def pick_choice_label(text: str) -> Optional[str]:
    match = ANSWER_LABEL_RE.search(text or "")
    return match.group(1).upper() if match else None


def display_answer_text(raw_value: str, option_lookup: Dict[str, str]) -> str:
    label = pick_choice_label(raw_value)
    if label and option_lookup.get(label):
        return f"{label}. {option_lookup[label]}"
    return raw_value.strip()


def grade_question(question: Dict[str, Any], kp: Dict[str, Any], user_answer: str) -> Dict[str, Any]:
    parsed_options = question.get("parsed_options") or []
    question_type = question.get("question_type") or "multiple_choice"
    option_lookup = {option["label"]: option["text"] for option in parsed_options if option.get("label")}
    review_refs = []
    for ref in question.get("review_refs") or []:
        source = str(ref.get("source") or "").replace("\\", "/")
        enriched = dict(ref)
        if source:
            enriched["material_url"] = url_for("course4186_material_asset", filename=source)
        review_refs.append(enriched)

    if not user_answer:
        reference = f"{correct_label}. {option_lookup.get(correct_label, question.get('answer', ''))}" if (correct_label := (question.get("correct_option") or "").upper().strip()) else question.get("answer", "")
        return {
            "status": "not_answered",
            "is_correct": False,
            "submitted_answer": "",
            "submitted_answer_display": "",
            "reference_answer": reference,
            "explanation": question.get("explanation") or "",
            "feedback_note": "No option was selected for this question.",
            "review_refs": review_refs,
        }

    if question_type != "multiple_choice":
        return {
            "status": "not_supported",
            "is_correct": False,
            "submitted_answer": user_answer,
            "submitted_answer_display": user_answer,
            "reference_answer": question.get("answer", ""),
            "explanation": question.get("explanation") or "",
            "feedback_note": "This question format is not available in the current student practice view.",
            "review_refs": review_refs,
        }

    correct_label = (question.get("correct_option") or "").upper().strip()
    answer_label = pick_choice_label(user_answer)
    answer_text = normalize_text(question.get("answer", ""))
    user_text = normalize_text(user_answer)

    is_correct = False
    if correct_label and answer_label == correct_label:
        is_correct = True
    elif correct_label and user_text == normalize_text(option_lookup.get(correct_label, "")):
        is_correct = True
    elif answer_text and user_text == answer_text:
        is_correct = True

    reference = f"{correct_label}. {option_lookup.get(correct_label, question.get('answer', ''))}" if correct_label else question.get("answer", "")
    return {
        "status": "correct" if is_correct else "incorrect",
        "is_correct": is_correct,
        "submitted_answer": user_answer,
        "submitted_answer_display": display_answer_text(user_answer, option_lookup),
        "reference_answer": reference,
        "explanation": question.get("explanation") or "",
        "feedback_note": (
            "Your choice matches the lecture conclusion for this question."
            if is_correct
            else "Review the explanation and lecture source below, then try this knowledge point again."
        ),
        "review_refs": review_refs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone Course 4186 student hub.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50186)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
