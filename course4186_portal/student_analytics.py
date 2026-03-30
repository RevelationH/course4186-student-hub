from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, Iterable, List, Sequence

from flask import url_for

from course4186_portal.kb_service import STOPWORDS, parse_options, tokenize

RELATED_KP_HINTS = {
    "kp-computer-vision-foundations": [
        "kp-image-filtering-and-convolution",
        "kp-edge-detection",
    ],
    "kp-image-filtering-and-convolution": [
        "kp-edge-detection",
        "kp-image-resampling",
        "kp-texture-analysis",
    ],
    "kp-edge-detection": [
        "kp-image-filtering-and-convolution",
        "kp-image-resampling",
        "kp-texture-analysis",
        "kp-harris-corners-and-local-features",
    ],
    "kp-image-resampling": [
        "kp-image-filtering-and-convolution",
        "kp-geometric-transformations",
    ],
    "kp-texture-analysis": [
        "kp-edge-detection",
        "kp-image-segmentation",
        "kp-bag-of-words-image-retrieval",
    ],
    "kp-image-segmentation": [
        "kp-edge-detection",
        "kp-texture-analysis",
    ],
    "kp-harris-corners-and-local-features": [
        "kp-sift-and-scale-invariant-features",
        "kp-image-alignment",
    ],
    "kp-sift-and-scale-invariant-features": [
        "kp-harris-corners-and-local-features",
        "kp-image-alignment",
        "kp-bag-of-words-image-retrieval",
    ],
    "kp-bag-of-words-image-retrieval": [
        "kp-sift-and-scale-invariant-features",
        "kp-harris-corners-and-local-features",
    ],
    "kp-geometric-transformations": [
        "kp-image-alignment",
        "kp-sift-and-scale-invariant-features",
        "kp-harris-corners-and-local-features",
    ],
    "kp-image-alignment": [
        "kp-geometric-transformations",
        "kp-harris-corners-and-local-features",
        "kp-sift-and-scale-invariant-features",
    ],
}


def build_dashboard_context(kb: Any, store: Any, user_id: str) -> Dict[str, Any]:
    state = _build_attempt_state(kb, store, user_id)
    kp_rows = []
    stats_map = {row["kp_id"]: row for row in state["kp_stats"]}
    for kp in kb.list_knowledge_points():
        stat = stats_map.get(kp["kp_id"], {})
        enriched = dict(kp)
        enriched["answered"] = stat.get("answered", 0)
        enriched["correct"] = stat.get("correct", 0)
        enriched["wrong"] = stat.get("wrong", 0)
        enriched["accuracy"] = stat.get("accuracy", 0.0)
        kp_rows.append(enriched)

    return {
        "summary": state["summary"],
        "kp_stats": state["kp_stats"],
        "kp_stats_map": stats_map,
        "kps": kp_rows,
        "recent_attempts": state["recent_attempts"],
        "latest_attempts": state["latest_attempts"],
    }


def build_learning_report_context(kb: Any, store: Any, user_id: str, follow_up_limit: int = 4) -> Dict[str, Any]:
    state = _build_attempt_state(kb, store, user_id)
    weak_points = _build_weak_points(state)
    strength_points = _build_strength_points(state)
    review_queue = _build_review_queue(state)
    recommendations = _build_recommendations(state, weak_points, strength_points)
    follow_up_questions = _build_follow_up_questions(kb, store, user_id, state, weak_points, limit=follow_up_limit)

    return {
        "summary": state["summary"],
        "kp_stats": state["kp_stats"],
        "weak_points": weak_points,
        "strength_points": strength_points,
        "review_queue": review_queue,
        "recommendations": recommendations,
        "recent_attempts": state["recent_attempts"],
        "follow_up_questions": follow_up_questions,
        "activity": state["activity"],
    }


def _build_attempt_state(kb: Any, store: Any, user_id: str) -> Dict[str, Any]:
    question_lookup = _question_lookup(kb)
    raw_attempts = store.all_attempts(user_id)
    filtered_attempts = _filter_current_attempts(raw_attempts, question_lookup)
    latest_attempts = _latest_attempts(filtered_attempts)
    kp_stats = _build_kp_stats(kb, latest_attempts)
    summary = _build_summary(latest_attempts, kp_stats, filtered_attempts)
    recent_attempts = _build_recent_attempts(filtered_attempts, limit=8)

    return {
        "question_lookup": question_lookup,
        "attempts": filtered_attempts,
        "latest_attempts": latest_attempts,
        "kp_stats": kp_stats,
        "summary": summary,
        "recent_attempts": recent_attempts,
        "activity": {
            "total_attempts": len(filtered_attempts),
            "recent_accuracy": _accuracy(filtered_attempts[-8:]),
        },
    }


def _question_lookup(kb: Any) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for questions in kb.questions_by_kp.values():
        for question in questions:
            lookup[str(question.get("question_id") or "")] = dict(question)
    return lookup


def _filter_current_attempts(
    attempts: Sequence[Dict[str, Any]],
    question_lookup: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for attempt in attempts:
        question_id = str(attempt.get("question_id") or "").strip()
        question = question_lookup.get(question_id)
        if not question:
            continue
        row = dict(attempt)
        row["kp_id"] = question.get("kp_id", attempt.get("kp_id"))
        row["kp_name"] = question.get("kp_name", attempt.get("kp_name"))
        row["question"] = question.get("question", attempt.get("question", ""))
        row["question_type"] = question.get("question_type", attempt.get("question_type"))
        row["submitted_answer"] = str(attempt.get("submitted_answer") or "").strip()
        row["reference_answer"] = question.get("correct_option") or attempt.get("reference_answer") or ""
        row["parsed_options"] = list(question.get("parsed_options") or [])
        row["is_correct"] = bool(attempt.get("is_correct"))
        row["explanation"] = question.get("explanation") or ""
        row["review_refs"] = _decorate_review_refs(question.get("review_refs") or [])
        row["image_path"] = question.get("image_path")
        row["image_caption"] = question.get("image_caption")
        row["timestamp_display"] = _format_timestamp(row.get("timestamp"))
        filtered.append(row)

    filtered.sort(key=lambda item: item.get("timestamp") or "")
    return filtered


def _latest_attempts(attempts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for attempt in attempts:
        key = str(attempt.get("question_id") or "")
        previous = latest.get(key)
        if previous is None or (attempt.get("timestamp") or "") >= (previous.get("timestamp") or ""):
            latest[key] = attempt
    rows = list(latest.values())
    rows.sort(key=lambda item: (item.get("timestamp") or "", item.get("question_id") or ""))
    return rows


def _build_summary(
    latest_attempts: Sequence[Dict[str, Any]],
    kp_stats: Sequence[Dict[str, Any]],
    all_attempts: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    answered = len(latest_attempts)
    correct = sum(1 for item in latest_attempts if item.get("is_correct"))
    wrong = answered - correct
    explored_points = sum(1 for row in kp_stats if row["answered"] > 0)
    mastered_points = sum(1 for row in kp_stats if row["answered"] >= 3 and row["accuracy"] >= 80)
    weak_points = sum(1 for row in kp_stats if row["answered"] > 0 and row["wrong"] > 0)
    recent_accuracy = _accuracy(all_attempts[-8:])
    overall_accuracy = round((correct / answered) * 100, 1) if answered else 0.0

    if len(all_attempts) < 3:
        trend_label = "Building history"
        trend_note = "Complete a few more sets to unlock a stronger trend signal."
    elif recent_accuracy >= overall_accuracy + 10:
        trend_label = "Improving"
        trend_note = "Your recent answers are stronger than your overall average."
    elif recent_accuracy <= overall_accuracy - 10:
        trend_label = "Needs attention"
        trend_note = "Recent answers have dipped below your overall level."
    else:
        trend_label = "Steady"
        trend_note = "Recent performance is close to your overall level."

    return {
        "answered": answered,
        "correct": correct,
        "wrong": wrong,
        "accuracy": overall_accuracy,
        "explored_points": explored_points,
        "mastered_points": mastered_points,
        "weak_points": weak_points,
        "recent_accuracy": recent_accuracy,
        "trend_label": trend_label,
        "trend_note": trend_note,
    }


def _build_kp_stats(kb: Any, latest_attempts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for kp in kb.list_knowledge_points():
        rows[kp["kp_id"]] = {
            "kp_id": kp["kp_id"],
            "kp_name": kp["name"],
            "description": kp["description"],
            "weeks": list(kp.get("weeks", [])),
            "answered": 0,
            "correct": 0,
            "wrong": 0,
            "accuracy": 0.0,
            "last_at": "",
        }

    for attempt in latest_attempts:
        kp_id = str(attempt.get("kp_id") or "")
        if kp_id not in rows:
            continue
        row = rows[kp_id]
        row["answered"] += 1
        if attempt.get("is_correct"):
            row["correct"] += 1
        else:
            row["wrong"] += 1
        row["last_at"] = max(row["last_at"], str(attempt.get("timestamp") or ""))

    for row in rows.values():
        row["accuracy"] = round((row["correct"] / row["answered"]) * 100, 1) if row["answered"] else 0.0

    return sorted(
        rows.values(),
        key=lambda row: (
            -(1 if row["answered"] > 0 else 0),
            -row["wrong"],
            row["accuracy"],
            row["kp_name"],
        ),
    )


def _build_weak_points(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    latest_attempts = state["latest_attempts"]
    wrong_latest = [item for item in latest_attempts if not item.get("is_correct")]
    wrong_by_kp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for attempt in wrong_latest:
        wrong_by_kp[str(attempt.get("kp_id") or "")].append(attempt)

    rows: List[Dict[str, Any]] = []
    for row in state["kp_stats"]:
        if row["answered"] == 0 or row["wrong"] == 0:
            continue
        mistakes = sorted(wrong_by_kp.get(row["kp_id"], []), key=lambda item: item.get("timestamp") or "", reverse=True)
        latest_mistake = mistakes[0] if mistakes else None
        entry = dict(row)
        entry["latest_mistake_question"] = latest_mistake.get("question") if latest_mistake else ""
        entry["latest_mistake_explanation"] = latest_mistake.get("explanation") if latest_mistake else ""
        entry["review_refs"] = list(latest_mistake.get("review_refs") or []) if latest_mistake else []
        rows.append(entry)

    rows.sort(key=lambda row: (-row["wrong"], row["accuracy"], -row["answered"], row["kp_name"]))
    return rows[:4]


def _build_strength_points(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [row for row in state["kp_stats"] if row["answered"] > 0 and row["accuracy"] >= 80]
    rows.sort(key=lambda row: (-row["accuracy"], -row["correct"], row["kp_name"]))
    return rows[:4]


def _build_review_queue(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [item for item in state["latest_attempts"] if not item.get("is_correct")]
    rows.sort(key=lambda item: item.get("timestamp") or "", reverse=True)
    return rows[:6]


def _build_recent_attempts(attempts: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    rows = list(attempts)[-limit:]
    rows.reverse()
    return rows


def _build_recommendations(
    state: Dict[str, Any],
    weak_points: Sequence[Dict[str, Any]],
    strength_points: Sequence[Dict[str, Any]],
) -> List[str]:
    recommendations: List[str] = []

    if weak_points:
        primary = weak_points[0]
        recommendations.append(
            f"Start with {primary['kp_name']}. Your current accuracy there is {primary['accuracy']}%, so it is the fastest place to gain marks."
        )
        if primary.get("review_refs"):
            recommendations.append(
                f"Review { _format_ref_brief(primary['review_refs'][0]) } first, then retry that topic before moving on."
            )

    if len(weak_points) > 1:
        secondary = weak_points[1]
        recommendations.append(
            f"Use a short second review block for {secondary['kp_name']}. The remaining mistakes there suggest the topic is still unstable under exam-style questions."
        )

    if strength_points:
        strongest = strength_points[0]
        recommendations.append(
            f"Keep {strongest['kp_name']} warm rather than over-studying it. It is already one of your strongest areas at {strongest['accuracy']}%."
        )

    if not recommendations:
        recommendations.append(
            "You do not have enough current quiz data yet. Start with one foundational set and come back after a few submissions."
        )

    recommendations.append(state["summary"]["trend_note"])
    return recommendations[:5]


def _build_follow_up_questions(
    kb: Any,
    store: Any,
    user_id: str,
    state: Dict[str, Any],
    weak_points: Sequence[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []

    answered_ids = {str(item.get("question_id") or "") for item in state["latest_attempts"]}
    used_ids: set[str] = set()
    selected: List[Dict[str, Any]] = []

    target_points = list(weak_points)
    if not target_points:
        target_points = [row for row in state["kp_stats"] if row["answered"] == 0][:2]

    variant_budget = _follow_up_variant_budget(kb, weak_points, limit)
    fixed_budget = max(limit - variant_budget, 0)

    for point in target_points:
        if len(selected) >= fixed_budget:
            break
        selected.extend(
            _pick_questions_from_kp(
                kb=kb,
                kp_id=point["kp_id"],
                answered_ids=answered_ids,
                used_ids=used_ids,
                limit=max(0, fixed_budget - len(selected)),
                reason=f"Recommended because {point['kp_name']} is currently a review priority.",
                focus_topic=point["kp_name"],
            )
        )
        if len(selected) >= fixed_budget:
            break

    if variant_budget > 0:
        variants = _build_llm_follow_up_questions(
            kb=kb,
            store=store,
            user_id=user_id,
            state=state,
            weak_points=weak_points,
            answered_ids=answered_ids,
            used_ids=used_ids,
            limit=min(variant_budget, limit - len(selected)),
        )
        selected.extend(variants)
        if len(selected) >= limit:
            return selected[:limit]

    for point in target_points:
        for related_kp_id in _related_kp_ids(kb, point["kp_id"]):
            selected.extend(
                _pick_questions_from_kp(
                    kb=kb,
                    kp_id=related_kp_id,
                    answered_ids=answered_ids,
                    used_ids=used_ids,
                    limit=1,
                    reason=f"Recommended as a follow-up to {point['kp_name']} because it builds on similar ideas.",
                    focus_topic=point["kp_name"],
                )
            )
            if len(selected) >= limit:
                return selected[:limit]

    for row in state["kp_stats"]:
        if row["answered"] == 0:
            selected.extend(
                _pick_questions_from_kp(
                    kb=kb,
                    kp_id=row["kp_id"],
                    answered_ids=answered_ids,
                    used_ids=used_ids,
                    limit=1,
                    reason="Recommended to broaden your coverage across the current full-course content.",
                    focus_topic=row["kp_name"],
                )
            )
            if len(selected) >= limit:
                return selected[:limit]

    weak_kp_ids = {str(item.get("kp_id") or "") for item in target_points}
    retry_candidates = [
        attempt
        for attempt in reversed(state["latest_attempts"])
        if not attempt.get("is_correct") and str(attempt.get("kp_id") or "") in weak_kp_ids
    ]
    for attempt in retry_candidates:
        question_id = str(attempt.get("question_id") or "")
        if not question_id or question_id in used_ids:
            continue
        used_ids.add(question_id)
        selected.append(
            _attempt_to_follow_up_item(
                attempt,
                reason=f"Recommended for a targeted retry because {attempt.get('kp_name') or 'this topic'} is still causing mistakes.",
                focus_topic=attempt.get("kp_name") or "Targeted Review",
            )
        )
        if len(selected) >= limit:
            return selected[:limit]

    return selected[:limit]


def _follow_up_variant_budget(kb: Any, weak_points: Sequence[Dict[str, Any]], limit: int) -> int:
    if limit <= 1 or not weak_points or not getattr(kb, "llm_enabled", False):
        return 0
    if limit >= 4 and len(weak_points) >= 2:
        return 2
    return 1


def _build_llm_follow_up_questions(
    kb: Any,
    store: Any,
    user_id: str,
    state: Dict[str, Any],
    weak_points: Sequence[Dict[str, Any]],
    answered_ids: set[str],
    used_ids: set[str],
    limit: int,
) -> List[Dict[str, Any]]:
    if limit <= 0 or not weak_points or not getattr(kb, "llm_enabled", False):
        return []

    cache_key = _follow_up_variant_cache_key(state, weak_points, limit)
    cached_rows = store.get_generated_followups(user_id, cache_key) if hasattr(store, "get_generated_followups") else []
    if cached_rows:
        rehydrated = _rehydrate_generated_follow_up_items(cached_rows)
        selected: List[Dict[str, Any]] = []
        for item in rehydrated:
            question_id = str(item.get("question_id") or "")
            if not question_id or question_id in used_ids or question_id in answered_ids:
                continue
            used_ids.add(question_id)
            selected.append(item)
            if len(selected) >= limit:
                return selected[:limit]

    rows_by_kp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for attempt in state["latest_attempts"]:
        if not attempt.get("is_correct"):
            rows_by_kp[str(attempt.get("kp_id") or "")].append(attempt)

    generated_rows: List[Dict[str, Any]] = []
    weak_targets = list(weak_points)[: max(limit, 2)]
    for point in weak_targets:
        if len(generated_rows) >= limit:
            break
        kp_id = str(point.get("kp_id") or "")
        if not kp_id:
            continue
        generated = kb.generate_follow_up_variants(
            kp_id,
            weak_attempts=rows_by_kp.get(kp_id, []),
            existing_questions=kb.related_questions(kp_id),
            question_count=1,
        )
        for item in generated:
            built = _build_generated_follow_up_item(
                item=item,
                kp_name=point.get("kp_name") or point.get("name") or "Targeted Review",
                unseen_count=_remaining_unseen_question_count(kb, kp_id, answered_ids),
            )
            if not built:
                continue
            generated_rows.append(built)
            if len(generated_rows) >= limit:
                break

    if generated_rows and hasattr(store, "set_generated_followups"):
        store.set_generated_followups(user_id, cache_key, generated_rows)

    selected: List[Dict[str, Any]] = []
    for item in _rehydrate_generated_follow_up_items(generated_rows):
        question_id = str(item.get("question_id") or "")
        if not question_id or question_id in used_ids or question_id in answered_ids:
            continue
        used_ids.add(question_id)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _follow_up_variant_cache_key(
    state: Dict[str, Any],
    weak_points: Sequence[Dict[str, Any]],
    limit: int,
) -> str:
    payload = {
        "limit": int(limit),
        "weak_points": [
            {
                "kp_id": row.get("kp_id"),
                "wrong": row.get("wrong"),
                "accuracy": row.get("accuracy"),
                "answered": row.get("answered"),
            }
            for row in weak_points[:4]
        ],
        "latest_attempts": [
            {
                "question_id": item.get("question_id"),
                "kp_id": item.get("kp_id"),
                "timestamp": item.get("timestamp"),
                "is_correct": bool(item.get("is_correct")),
            }
            for item in state["latest_attempts"]
        ],
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return f"llm-followups-{digest[:16]}"


def _build_generated_follow_up_item(
    item: Dict[str, Any],
    *,
    kp_name: str,
    unseen_count: int,
) -> Dict[str, Any] | None:
    question_id = str(item.get("question_id") or "").strip()
    if not question_id:
        return None
    reason = (
        f"Recommended as a fresh variant for {kp_name} so you can test the same idea with a new exam-style prompt."
        if unseen_count > 0 else
        f"Recommended as a fresh variant because you have already worked through the current fixed questions in {kp_name}."
    )
    return {
        "question_id": question_id,
        "kp_id": item.get("kp_id"),
        "kp_name": item.get("kp_name") or kp_name,
        "question": item.get("question"),
        "parsed_options": [dict(option) for option in item.get("parsed_options") or []],
        "options": list(item.get("options") or []),
        "reference_answer": item.get("reference_answer") or _format_reference_answer(item),
        "image_path": item.get("image_path"),
        "image_caption": item.get("image_caption"),
        "explanation": item.get("explanation") or "",
        "review_refs": list(item.get("review_refs") or []),
        "reason": reason,
        "focus_topic": kp_name,
        "generated": True,
    }


def _rehydrate_generated_follow_up_items(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hydrated: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if not item.get("parsed_options") and item.get("options"):
            item["parsed_options"] = parse_options(item.get("options") or [])
        else:
            item["parsed_options"] = [dict(option) for option in item.get("parsed_options") or []]
        item["review_refs"] = _decorate_review_refs(item.get("review_refs") or [])
        hydrated.append(item)
    return hydrated


def _remaining_unseen_question_count(kb: Any, kp_id: str, answered_ids: set[str]) -> int:
    total = 0
    for question in kb.related_questions(kp_id):
        question_id = str(question.get("question_id") or "")
        if question_id and question_id not in answered_ids:
            total += 1
    return total


def _pick_questions_from_kp(
    kb: Any,
    kp_id: str,
    answered_ids: set[str],
    used_ids: set[str],
    limit: int,
    reason: str,
    focus_topic: str,
) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    rows: List[Dict[str, Any]] = []
    questions = list(kb.related_questions(kp_id))
    questions.sort(key=lambda item: str(item.get("question_id") or ""))
    for question in questions:
        question_id = str(question.get("question_id") or "")
        if not question_id or question_id in used_ids or question_id in answered_ids:
            continue
        used_ids.add(question_id)
        rows.append(
            {
                "question_id": question_id,
                "kp_id": question.get("kp_id"),
                "kp_name": question.get("kp_name"),
                "question": question.get("question"),
                "parsed_options": list(question.get("parsed_options") or []),
                "reference_answer": _format_reference_answer(question),
                "image_path": question.get("image_path"),
                "image_caption": question.get("image_caption"),
                "explanation": question.get("explanation") or "",
                "review_refs": _decorate_review_refs(question.get("review_refs") or []),
                "reason": reason,
                "focus_topic": focus_topic,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _attempt_to_follow_up_item(attempt: Dict[str, Any], reason: str, focus_topic: str) -> Dict[str, Any]:
    return {
        "question_id": attempt.get("question_id"),
        "kp_id": attempt.get("kp_id"),
        "kp_name": attempt.get("kp_name"),
        "question": attempt.get("question"),
        "parsed_options": list(attempt.get("parsed_options") or []),
        "reference_answer": _format_reference_answer(attempt),
        "image_path": attempt.get("image_path"),
        "image_caption": attempt.get("image_caption"),
        "explanation": attempt.get("explanation") or "",
        "review_refs": list(attempt.get("review_refs") or []),
        "reason": reason,
        "focus_topic": focus_topic,
    }


def _related_kp_ids(kb: Any, kp_id: str) -> List[str]:
    kp = kb.kp_by_id.get(kp_id)
    if not kp:
        return []

    related_ids: List[str] = []
    for candidate_id in RELATED_KP_HINTS.get(kp_id, []):
        if candidate_id in kb.kp_by_id and candidate_id != kp_id and candidate_id not in related_ids:
            related_ids.append(candidate_id)

    if len(related_ids) >= 4:
        return related_ids[:4]

    source_tokens = _kp_tokens(kp)
    fallback_scores: List[tuple[float, str]] = []
    source_week = _first_week_number(kp.get("weeks", []))
    for candidate in kb.kps:
        candidate_id = str(candidate.get("kp_id") or "")
        if not candidate_id or candidate_id == kp_id or candidate_id in related_ids:
            continue
        candidate_tokens = _kp_tokens(candidate)
        overlap = len(source_tokens & candidate_tokens)
        if overlap == 0:
            continue
        week_gap = abs((_first_week_number(candidate.get("weeks", [])) or 99) - (source_week or 99))
        score = overlap * 3 - week_gap * 0.4
        fallback_scores.append((score, candidate_id))

    fallback_scores.sort(key=lambda item: item[0], reverse=True)
    for _, candidate_id in fallback_scores:
        if candidate_id not in related_ids:
            related_ids.append(candidate_id)
        if len(related_ids) >= 4:
            break
    return related_ids[:4]


def _kp_tokens(kp: Dict[str, Any]) -> set[str]:
    tokens = tokenize(
        " ".join(
            [
                str(kp.get("name") or ""),
                str(kp.get("description") or ""),
                " ".join(kp.get("keywords", [])),
            ]
        )
    )
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def _first_week_number(weeks: Iterable[str]) -> int | None:
    for week in weeks:
        text = str(week or "").strip().lower()
        if text.startswith("week"):
            digits = "".join(char for char in text if char.isdigit())
            if digits:
                return int(digits)
    return None


def _accuracy(rows: Sequence[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    correct = sum(1 for item in rows if item.get("is_correct"))
    return round((correct / len(rows)) * 100, 1)


def _format_timestamp(raw_value: Any) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return "No timestamp"
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return text


def _format_ref_brief(ref: Dict[str, Any]) -> str:
    display_source = str(ref.get("display_source") or ref.get("source") or "lecture material").replace("\\", "/")
    location = str(ref.get("location") or "").strip()
    if location:
        return f"{display_source} {location}"
    return display_source


def _format_reference_answer(question: Dict[str, Any]) -> str:
    correct_option = str(question.get("correct_option") or "").strip().upper()
    parsed_options = list(question.get("parsed_options") or [])
    option_lookup = {str(option.get("label") or "").upper(): str(option.get("text") or "") for option in parsed_options}
    if correct_option and option_lookup.get(correct_option):
        return f"{correct_option}. {option_lookup[correct_option]}"
    return str(question.get("answer") or "").strip()


def _decorate_review_refs(review_refs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ref in review_refs:
        item = dict(ref)
        source = str(item.get("source") or "").replace("\\", "/")
        if source:
            params = {"source": source}
            if item.get("unit_type"):
                params["unit_type"] = item.get("unit_type")
            if item.get("unit_index") not in (None, ""):
                params["unit_index"] = item.get("unit_index")
            if item.get("chunk_index") not in (None, ""):
                params["chunk_index"] = item.get("chunk_index")
            item["material_url"] = url_for("course4186_material_open", **params)
            item["reference_url"] = url_for("course4186_material_reference", **params)
        rows.append(item)
    return rows
