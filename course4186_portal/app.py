from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, session, url_for


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from course4186_portal.kb_service import Course4186KnowledgeBase, STOPWORDS
from course4186_portal.progress_store import ProgressStore

try:
    from user import User
    from werkzeug.security import check_password_hash
    AUTH_BACKEND_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - defensive import fallback
    User = None
    check_password_hash = None
    AUTH_BACKEND_ERROR = exc


ANSWER_LABEL_RE = re.compile(r"\b([A-D])\b", flags=re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*")
PUBLIC_ENDPOINTS = {"healthz", "login_4186", "logout_4186", "course4186_web_asset", "static"}


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


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(APP_DIR / "templates"))
    app.config["SECRET_KEY"] = os.getenv("COURSE4186_PORTAL_SECRET", "course4186-portal-secret")
    app.config["SESSION_COOKIE_NAME"] = "course4186_portal_session"
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    kb = Course4186KnowledgeBase()
    store = ProgressStore(APP_DIR / "data" / "progress.json")

    app.config["kb"] = kb
    app.config["store"] = store

    def render_login_page(
        *,
        error_message: str = "",
        username_value: str = "",
        next_path: str = "",
        status_code: int = 200,
    ) -> Any:
        resolved_next = safe_next_path(next_path) or url_for("chatapi_4186")
        return (
            render_template(
                "login_4186.html",
                error_message=error_message,
                username_value=username_value,
                next_path=resolved_next,
                auth_backend_ready=AUTH_BACKEND_ERROR is None,
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
        summary = store.summary(user_id) if user_id else {"answered": 0, "accuracy": 0.0}
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
            return render_login_page(next_path=request.args.get("next", ""))

        username = clean_login_name(request.form.get("username"))
        password = request.form.get("password") or ""
        next_path = request.form.get("next") or ""

        if AUTH_BACKEND_ERROR is not None or User is None or check_password_hash is None:
            return render_login_page(
                error_message=login_error_message(AUTH_BACKEND_ERROR or RuntimeError("Unknown backend error")),
                username_value=username,
                next_path=next_path,
                status_code=503,
            )

        if not username or not password:
            return render_login_page(
                error_message="Please enter both username and password.",
                username_value=username,
                next_path=next_path,
                status_code=400,
            )

        try:
            user = User.get_by_username(username)
        except Exception as exc:
            return render_login_page(
                error_message=login_error_message(exc),
                username_value=username,
                next_path=next_path,
                status_code=503,
            )

        if not user or not check_password_hash(user.password, password):
            return render_login_page(
                error_message="Incorrect username or password.",
                username_value=username,
                next_path=next_path,
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
                "artifact_dir": str(kb.artifact_dir),
                "llm_enabled": bool(kb._llm_client and kb._llm_model),
                "llm_model": kb._llm_model,
                "dense_enabled": kb.dense_enabled,
                "knowledge_points": len(kb.kps),
                "questions": len(kb.questions),
                "chunks": len(kb.chunks),
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
        history = payload.get("history") or []
        if not message:
            return jsonify({"ok": False, "error": "Please enter a question before sending."}), 400

        answer = kb.answer(message, history=history, top_k=5)
        return jsonify(
            {
                "ok": True,
                "answer": answer["answer"],
                "citations": answer["citations"],
                "related_kps": answer["related_kps"],
                "mode": answer["mode"],
                "actions": {
                    "quiz_url": url_for("quiz_dashboard_4186"),
                    "report_url": url_for("quiz_analysis_4186"),
                },
            }
        )

    @app.get("/quiz_4186")
    def quiz_dashboard_4186() -> Any:
        kp_stats_map = {row["kp_id"]: row for row in store.kp_stats(session["course4186_user_id"])}
        kps = []
        for row in kb.list_knowledge_points():
            stat = kp_stats_map.get(row["kp_id"], {})
            enriched = dict(row)
            enriched["answered"] = stat.get("answered", 0)
            enriched["correct"] = stat.get("correct", 0)
            enriched["wrong"] = stat.get("wrong", 0)
            enriched["accuracy"] = stat.get("accuracy", 0.0)
            kps.append(enriched)
        return render_template("quiz_dashboard_4186.html", kps=kps, profile=store.get_user(session["course4186_user_id"]))

    @app.route("/quiz_4186/practice/<string:kp_id>", methods=["GET", "POST"])
    def quiz_practice_4186(kp_id: str) -> Any:
        kp = kb.get_kp(kp_id)
        if not kp:
            return "Knowledge point not found.", 404

        questions = kp.get("questions", [])
        results_map: Dict[str, Dict[str, Any]] = {}
        summary = None
        grading_note = "This 4186 practice set contains only multiple-choice questions and uses exact answer grading."

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
            grading_note=grading_note,
        )

    @app.get("/quiz_4186/analysis")
    def quiz_analysis_4186() -> Any:
        user_id = session["course4186_user_id"]
        summary = store.summary(user_id)
        kp_stats = store.kp_stats(user_id)
        weak_points = store.weak_points(user_id)
        recommendations = kb.recommendation_for_weak_points(weak_points)
        recent_attempts = store.recent_attempts(user_id, limit=10)
        return render_template(
            "quiz_analysis_4186.html",
            summary=summary,
            kp_stats=kp_stats,
            weak_points=weak_points,
            recommendations=recommendations,
            recent_attempts=recent_attempts,
            profile=store.get_user(user_id),
        )

    return app


def normalize_text(text: str) -> str:
    compact = re.sub(r"[^a-z0-9\s]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", compact).strip()


def pick_choice_label(text: str) -> Optional[str]:
    match = ANSWER_LABEL_RE.search(text or "")
    return match.group(1).upper() if match else None


def extract_keywords(text: str, limit: int = 10) -> List[str]:
    counts = Counter(
        token.lower()
        for token in TOKEN_RE.findall(text or "")
        if len(token) > 2 and token.lower() not in STOPWORDS
    )
    return [token for token, _ in counts.most_common(limit)]


def grade_question(question: Dict[str, Any], kp: Dict[str, Any], user_answer: str) -> Dict[str, Any]:
    parsed_options = question.get("parsed_options") or []
    question_type = question.get("question_type") or "multiple_choice"

    if not user_answer:
        return {
            "is_correct": False,
            "submitted_answer": "",
            "reference_answer": question.get("correct_option") or question.get("answer"),
            "explanation": question.get("explanation") or "",
            "grading_note": "No answer submitted.",
            "matched_keywords": [],
        }

    if question_type != "multiple_choice":
        return {
            "is_correct": False,
            "submitted_answer": user_answer,
            "reference_answer": question.get("answer", ""),
            "explanation": question.get("explanation") or "",
            "grading_note": "This student hub currently supports only multiple-choice questions.",
            "matched_keywords": [],
        }

    correct_label = (question.get("correct_option") or "").upper().strip()
    answer_label = pick_choice_label(user_answer)
    answer_text = normalize_text(question.get("answer", ""))
    user_text = normalize_text(user_answer)
    option_lookup = {option["label"]: option["text"] for option in parsed_options if option.get("label")}

    is_correct = False
    if correct_label and answer_label == correct_label:
        is_correct = True
    elif correct_label and user_text == normalize_text(option_lookup.get(correct_label, "")):
        is_correct = True
    elif answer_text and user_text == answer_text:
        is_correct = True

    reference = f"{correct_label}. {option_lookup.get(correct_label, question.get('answer', ''))}" if correct_label else question.get("answer", "")
    return {
        "is_correct": is_correct,
        "submitted_answer": user_answer,
        "reference_answer": reference,
        "explanation": question.get("explanation") or "",
        "grading_note": "Multiple-choice answers are graded against the standard answer.",
        "matched_keywords": [],
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
