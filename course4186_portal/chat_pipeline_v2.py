from __future__ import annotations

import json
import os
import re
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from course4186_portal.kb_service import (
    Course4186KnowledgeBase,
    clean_display_text,
    contains_transport_artifacts,
    normalized_path_key,
    safe_snippet,
    source_family_key,
)

GENERIC_NON_TOPIC_CONCEPT_LABELS = {
    "example",
    "examples",
    "sample",
    "samples",
    "building",
    "build",
    "illustrate",
    "illustration",
    "using",
    "show",
    "write",
    "code",
    "snippet",
    "implementation",
    "implement",
}
CODE_REQUEST_NON_TOPIC_CONCEPT_LABELS = GENERIC_NON_TOPIC_CONCEPT_LABELS | {
    "python",
    "pytorch",
    "torch",
    "c++",
    "cpp",
    "java",
    "javascript",
    "js",
}
ROOT_DIR = Path(__file__).resolve().parents[1]
DISPLAY_SOURCE_RECAP_RE = re.compile(r"\b(revision|recap|summary)\b", re.IGNORECASE)
DISPLAY_SOURCE_LECTURE_RE = re.compile(r"(^|[\\/])lecture", re.IGNORECASE)
DISPLAY_SOURCE_GENERIC_TITLE_RE = re.compile(r"^(topics?|summary)$", re.IGNORECASE)


@lru_cache(maxsize=8)
def _course_material_root_candidates() -> tuple[Path, ...]:
    candidates: List[Path] = []
    env_root = os.getenv("COURSE4186_MATERIAL_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            ROOT_DIR / "4186" / "4186",
            ROOT_DIR / "course4186_materials",
            ROOT_DIR / "course4186_portal" / "materials",
        ]
    )
    resolved: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        except Exception:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(candidate)
    return tuple(resolved)


@lru_cache(maxsize=1024)
def _material_exists(relative_path: str) -> bool:
    normalized = normalized_path_key(relative_path)
    if not normalized:
        return False
    parts = [part for part in normalized.split("/") if part and part not in {".", ".."}]
    if not parts:
        return False
    target = Path(*parts)
    for root in _course_material_root_candidates():
        candidate = root / target
        if candidate.exists() and candidate.is_file():
            return True

    target_name = target.name.lower()
    target_stem = target.stem.lower()
    for root in _course_material_root_candidates():
        if not root.exists():
            continue
        try:
            matches = root.rglob(target.name)
        except Exception:
            continue
        for candidate in matches:
            if not candidate.is_file():
                continue
            if candidate.name.lower() == target_name:
                return True
        if not target_name.endswith(".pdf"):
            continue
        for candidate in root.rglob("*.pdf"):
            if candidate.is_file() and candidate.stem.lower() == target_stem:
                return True
    return False


class Course4186ChatPipeline:
    def __init__(self, kb: Course4186KnowledgeBase) -> None:
        self.kb = kb

    def _request_preferences(self, raw_query: str, resolved_query: str) -> Dict[str, str]:
        combined = " ".join(
            clean_display_text(part).lower()
            for part in (raw_query, resolved_query)
            if clean_display_text(part)
        )
        language = self.kb._query_language_hint(raw_query) or self.kb._query_language_hint(resolved_query)
        framework = ""
        if "pytorch" in combined or re.search(r"\btorch\b", combined):
            framework = "PyTorch"
            if not language:
                language = "Python"
        elif "opencv" in combined or "cv2" in combined:
            framework = "OpenCV"
        return {
            "language": language,
            "framework": framework,
        }

    def _concept_is_output_or_instruction_noise(self, concept: Dict[str, Any], *, code_request: bool) -> bool:
        label = clean_display_text(concept.get("label", "")).lower()
        tokens = {
            clean_display_text(token).lower()
            for token in concept.get("tokens", set())
            if clean_display_text(token)
        }
        blocked = CODE_REQUEST_NON_TOPIC_CONCEPT_LABELS if code_request else GENERIC_NON_TOPIC_CONCEPT_LABELS
        if label in blocked:
            return True
        if tokens and tokens.issubset(blocked):
            return True
        return False

    def _sanitize_target_concepts(
        self,
        concepts: Sequence[Dict[str, Any]],
        *,
        code_request: bool,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen_labels: set[str] = set()
        for concept in concepts or []:
            if self._concept_is_output_or_instruction_noise(concept, code_request=code_request):
                continue
            label = clean_display_text(concept.get("label", ""))
            if not label:
                continue
            lowered = label.lower()
            if lowered in seen_labels:
                continue
            seen_labels.add(lowered)
            rows.append(
                {
                    "label": label,
                    "phrases": set(concept.get("phrases", set()) or {label.lower()}),
                    "tokens": set(concept.get("tokens", set()) or self.kb._concept_tokens(label)),
                    "position": int(concept.get("position", 10**6) or 10**6),
                    "source": clean_display_text(concept.get("source", "")) or "phrase",
                }
            )
        rows.sort(key=lambda item: (int(item.get("position", 10**6)), clean_display_text(item.get("label", "")).lower()))
        return rows

    def _sanitize_task_profile(
        self,
        raw_query: str,
        resolved_query: str,
        task_profile: Dict[str, Any],
        kp_hits: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        normalized = dict(task_profile or {})
        request_preferences = self._request_preferences(raw_query, resolved_query)
        code_request = bool(normalized.get("is_code_request"))

        sanitized_concepts = self._sanitize_target_concepts(
            normalized.get("target_concepts") or [],
            code_request=code_request,
        )
        if not sanitized_concepts:
            sanitized_concepts = self._sanitize_target_concepts(
                self.kb._query_target_concepts(resolved_query, kp_hits, limit=4),
                code_request=code_request,
            )
        normalized["target_concepts"] = sanitized_concepts
        normalized["request_preferences"] = request_preferences

        if code_request and len(sanitized_concepts) <= 1:
            normalized["needs_multi_concept_coverage"] = False

        if code_request and not sanitized_concepts and normalized.get("relevant_kp_hits"):
            normalized["needs_multi_concept_coverage"] = False

        return normalized

    def _build_related_topics(
        self,
        task_profile: Dict[str, Any],
        kp_hits: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        source_hits = list(task_profile.get("relevant_kp_hits") or [])
        if not source_hits:
            source_hits = list(kp_hits[:3])

        rows: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        for hit in source_hits:
            item = dict(hit.get("item") or {})
            kp_id = clean_display_text(item.get("kp_id", ""))
            if not kp_id or kp_id in seen_ids:
                continue
            seen_ids.add(kp_id)
            rows.append(
                {
                    "kp_id": kp_id,
                    "name": clean_display_text(item.get("name", "")),
                    "description": clean_display_text(item.get("description", "")),
                }
            )
            if len(rows) >= 3:
                break
        return rows

    def _retrieve_context(
        self,
        raw_query: str,
        resolved_query: str,
        *,
        top_k: int,
    ) -> Dict[str, Any]:
        kp_hits = self.kb.search_knowledge_points(resolved_query, top_k=max(8, top_k + 3))
        task_profile = self.kb._query_task_profile(resolved_query, kp_hits)
        raw_intent_profile = self.kb._query_task_profile(raw_query, kp_hits)

        if raw_intent_profile.get("is_code_request"):
            task_profile["is_code_request"] = True
            task_profile["code_request_mode"] = raw_intent_profile.get("code_request_mode", "code_only")
            task_profile["is_code_only_request"] = bool(raw_intent_profile.get("is_code_only_request"))
            task_profile["is_code_plus_explanation_request"] = bool(raw_intent_profile.get("is_code_plus_explanation_request"))

        if raw_intent_profile.get("is_question_generation_request"):
            task_profile["is_question_generation_request"] = True
            task_profile["practice_question_mode"] = raw_intent_profile.get("practice_question_mode", "none")

        task_profile = self._sanitize_task_profile(raw_query, resolved_query, task_profile, kp_hits)

        chunk_search_k = max(8, top_k * 2) if task_profile.get("needs_multi_concept_coverage") else max(6, top_k + 2)
        chunk_hits = self.kb.search_chunks(resolved_query, top_k=chunk_search_k, kp_context=kp_hits)
        focused_hits = self.kb._focused_kp_support_hits(
            resolved_query,
            kp_hits,
            limit=6,
            task_profile=task_profile,
        )
        multi_concept_hits = (
            self.kb._multi_concept_chunk_hits(
                resolved_query,
                chunk_hits,
                limit=4,
                task_profile=task_profile,
            )
            if task_profile.get("needs_multi_concept_coverage")
            else []
        )

        if multi_concept_hits:
            citation_hits = multi_concept_hits
        else:
            preferred_candidates = focused_hits if len(focused_hits) >= 2 else chunk_hits
            citation_hits = self.kb._preferred_chunk_hits(
                resolved_query,
                preferred_candidates,
                limit=4,
                kp_context=kp_hits,
            )
            if not citation_hits and chunk_hits:
                citation_hits = list(chunk_hits[:4])

        citation_hits = self._filter_display_citation_hits(
            resolved_query,
            task_profile=task_profile,
            citation_hits=citation_hits,
            chunk_hits=chunk_hits,
        )
        citations = self.kb._build_citations(citation_hits, limit=4)
        coverage_level = self.kb._course_coverage_level(
            resolved_query,
            citation_hits,
            task_profile=task_profile,
        )
        related_kps = self._build_related_topics(task_profile, kp_hits)

        return {
            "kp_hits": kp_hits,
            "task_profile": task_profile,
            "chunk_hits": chunk_hits,
            "citation_hits": citation_hits,
            "citations": citations,
            "coverage_level": coverage_level,
            "related_kps": related_kps,
        }

    def _filter_display_citation_hits(
        self,
        resolved_query: str,
        *,
        task_profile: Dict[str, Any],
        citation_hits: Sequence[Dict[str, Any]],
        chunk_hits: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        target_concepts = list(task_profile.get("target_concepts") or [])
        definition_request = bool(task_profile.get("is_definition"))

        candidates: List[Dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        for hit in list(citation_hits or []) + list(chunk_hits or [])[:10]:
            item = hit.get("item") or {}
            chunk_id = clean_display_text(item.get("chunk_id", ""))
            if not chunk_id or chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            candidates.append(hit)

        scored_rows: List[Dict[str, Any]] = []
        for hit in candidates:
            item = hit.get("item") or {}
            relative_path = clean_display_text(item.get("relative_path", ""))
            if relative_path and not _material_exists(relative_path):
                continue
            matched_labels: List[str] = []
            best_score = float(hit.get("score", 0.0))
            for concept in target_concepts:
                if not self.kb._concept_matches_chunk_item(concept, item):
                    continue
                label = clean_display_text(concept.get("label", ""))
                if label:
                    matched_labels.append(label)
                best_score = max(best_score, self.kb._concept_chunk_match_score(concept, item, float(hit.get("score", 0.0))))
            if target_concepts and not matched_labels:
                continue
            title_text = clean_display_text(item.get("title", ""))
            family = source_family_key(relative_path)
            source_name = Path(relative_path).name.lower() if relative_path else ""
            scored_rows.append(
                {
                    "hit": hit,
                    "labels": matched_labels,
                    "score": best_score,
                    "family": family,
                    "chunk_id": clean_display_text(item.get("chunk_id", "")),
                    "unit_index": int(item.get("unit_index", 0) or 0) if str(item.get("unit_index", "")).strip() else 0,
                    "is_low_priority": self.kb._is_low_priority_source(item),
                    "is_recap_like": bool(
                        DISPLAY_SOURCE_RECAP_RE.search(relative_path)
                        or DISPLAY_SOURCE_RECAP_RE.search(title_text)
                        or Path(relative_path).stem.lower() == "all"
                    ),
                    "is_lecture_like": bool(DISPLAY_SOURCE_LECTURE_RE.search(relative_path or source_name)),
                    "is_generic_title": bool(DISPLAY_SOURCE_GENERIC_TITLE_RE.fullmatch(title_text.strip())),
                }
            )

        if not scored_rows:
            fallback_rows: List[Dict[str, Any]] = []
            for hit in candidates:
                item = hit.get("item") or {}
                relative_path = clean_display_text(item.get("relative_path", ""))
                if relative_path and not _material_exists(relative_path):
                    continue
                fallback_rows.append(hit)
            return list(fallback_rows[:3] or citation_hits[:3])

        scored_rows.sort(
            key=lambda row: (
                row["is_low_priority"],
                row["is_recap_like"],
                0 if row["is_lecture_like"] else 1,
                row["is_generic_title"],
                -len(row["labels"]),
                -float(row["score"]),
                row["unit_index"] if definition_request and row["unit_index"] > 0 else 10**6,
            )
        )

        selected: List[Dict[str, Any]] = []
        selected_chunk_ids: set[str] = set()
        selected_families: set[str] = set()
        covered_labels: set[str] = set()

        def _pick_row(
            rows: Sequence[Dict[str, Any]],
            *,
            allow_same_family: bool = False,
        ) -> Optional[Dict[str, Any]]:
            for row in rows:
                chunk_id = clean_display_text(row.get("chunk_id", ""))
                family = clean_display_text(row.get("family", ""))
                if not chunk_id or chunk_id in selected_chunk_ids:
                    continue
                if family and family in selected_families and not allow_same_family:
                    continue
                return row
            return None

        for concept in target_concepts[:3]:
            concept_label = clean_display_text(concept.get("label", ""))
            if not concept_label or concept_label.lower() in covered_labels:
                continue
            concept_rows = [row for row in scored_rows if concept_label in row["labels"]]
            best_row = _pick_row(concept_rows)
            if not best_row:
                best_row = _pick_row(
                    [row for row in concept_rows if not row["is_low_priority"] and not row["is_recap_like"]],
                    allow_same_family=True,
                )
            if not best_row:
                best_row = _pick_row(concept_rows, allow_same_family=True)
            if not best_row:
                continue
            item = best_row["hit"]["item"]
            selected.append(best_row["hit"])
            selected_chunk_ids.add(clean_display_text(item.get("chunk_id", "")))
            family = clean_display_text(best_row.get("family", ""))
            if family:
                selected_families.add(family)
            covered_labels.add(concept_label.lower())

        max_results = 3
        if target_concepts:
            max_results = 2 if len(target_concepts) == 1 else min(3, len(target_concepts))
            if len(target_concepts) >= 2 and selected:
                return selected[:max_results]

        for allow_same_family in (False, True):
            for row in scored_rows:
                picked = _pick_row([row], allow_same_family=allow_same_family)
                if not picked:
                    continue
                item = picked["hit"]["item"]
                selected.append(picked["hit"])
                selected_chunk_ids.add(clean_display_text(item.get("chunk_id", "")))
                family = clean_display_text(picked.get("family", ""))
                if family:
                    selected_families.add(family)
                if len(selected) >= max_results:
                    return selected[:max_results]

        return selected[:max_results] if selected else list(citation_hits[:max_results])

    def _history_text(self, history: Sequence[Dict[str, str]], limit: int = 8) -> str:
        rows: List[str] = []
        for turn in list(history)[-limit:]:
            role = clean_display_text(turn.get("role", "")) or "user"
            content = clean_display_text(turn.get("content", ""))
            if not content:
                continue
            rows.append(f"{role}: {content}")
        return "\n".join(rows) or "None"

    def _evidence_text(
        self,
        citation_hits: Sequence[Dict[str, Any]],
        citations: Sequence[Dict[str, Any]],
    ) -> str:
        rows: List[str] = []
        for index, citation in enumerate(list(citations or [])[:4]):
            item = citation_hits[index].get("item", {}) if index < len(citation_hits) else {}
            source_label = clean_display_text(citation.get("display_source") or citation.get("source") or "")
            location = clean_display_text(citation.get("location") or "")
            section = clean_display_text(citation.get("section") or item.get("title", "")) or "Untitled section"
            excerpt = safe_snippet(clean_display_text(item.get("text", "")), 520) or "None"
            rows.append(
                "\n".join(
                    [
                        f"[{citation['citation_id']}] {source_label} ({location or 'source'})",
                        f"Section: {section}",
                        f"Excerpt: {excerpt}",
                    ]
                ).strip()
            )
        return "\n\n".join(rows) or "None"

    def _allowed_source_ids(self, citations: Sequence[Dict[str, Any]]) -> List[str]:
        rows: List[str] = []
        for item in citations:
            source_id = clean_display_text(item.get("citation_id", "")).upper()
            if source_id and source_id not in rows:
                rows.append(source_id)
        return rows

    def _normalize_payload(
        self,
        payload: Dict[str, Any],
        *,
        query: str,
        task_profile: Dict[str, Any],
        related_kps: Sequence[Dict[str, Any]],
        citation_hits: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        normalized = dict(payload or {})
        answer = str(normalized.get("answer") or "").strip()
        if task_profile.get("is_code_request"):
            answer = self.kb._ensure_code_answer_explanation(
                answer,
                query,
                related_kps,
                citation_hits,
                code_request_mode=clean_display_text(task_profile.get("code_request_mode", "")) or "code_plus_explanation",
            )
            if task_profile.get("is_code_only_request"):
                normalized["used_sources"] = []
        elif self.kb._answer_contains_code_block(answer) and not self.kb._code_block_quality_ok(answer, query):
            prose_only = self.kb._strip_code_blocks_for_validation(answer).strip()
            prose_only = re.sub(
                r"(?is)\s*(?:here(?:'s| is)|below is|the following is)\b[^.?!]*\b(code|snippet|example)\b[^.?!]*[:.]?\s*$",
                "",
                prose_only,
            ).strip()
            if self.kb._meaningful_prose_word_count(prose_only) >= 12:
                answer = prose_only
        normalized["answer"] = answer
        return normalized

    def _answer_issues(
        self,
        payload: Dict[str, Any],
        *,
        resolved_query: str,
        task_profile: Dict[str, Any],
        allowed_source_ids: Sequence[str],
        coverage_level: str,
        citations: Sequence[Dict[str, Any]],
        recent_answers: Sequence[str],
    ) -> List[str]:
        answer = str(payload.get("answer") or "").strip()
        used_sources = [
            clean_display_text(item).upper()
            for item in payload.get("used_sources", [])
            if clean_display_text(item)
        ]
        allowed = set(allowed_source_ids)
        issues: List[str] = []

        if not answer:
            return ["The answer is empty."]
        if contains_transport_artifacts(answer):
            issues.append("The answer still contains transport artifacts or source placeholders.")
        if any(source_id not in allowed for source_id in used_sources):
            issues.append("The answer selected source ids that are not in the supplied evidence.")
        if coverage_level == "none" and used_sources:
            issues.append("The answer should not cite course sources when the retrieved evidence does not support the topic.")
        if coverage_level == "direct" and citations and not used_sources and not task_profile.get("is_code_only_request"):
            issues.append("The retrieved course evidence directly supports this topic, so keep at least one genuinely supporting source id.")
        if not self.kb._answer_satisfies_task(answer, resolved_query, task_profile):
            issues.append("The answer does not fully satisfy the student's actual request.")
        if self.kb._answer_contains_code_block(answer) and not self.kb._code_block_quality_ok(answer, resolved_query):
            issues.append("Any included Markdown code block must be complete and well-formed.")

        if task_profile.get("is_code_request"):
            has_code_block = self.kb._answer_contains_code_block(answer)
            prose_outside_code = self.kb._strip_code_blocks_for_validation(answer)
            prose_word_count = self.kb._meaningful_prose_word_count(prose_outside_code)
            if not has_code_block:
                issues.append("The student asked for code, but the answer does not contain a fenced Markdown code block.")
            elif task_profile.get("is_code_only_request") and prose_word_count > 0:
                issues.append("This is a code-only request, so there must be no prose outside the code block.")
            elif task_profile.get("is_code_plus_explanation_request") and prose_word_count < 6:
                issues.append("This request asks for both explanation and code, so the answer needs a short explanation plus the code block.")
            if has_code_block and not self.kb._code_block_quality_ok(answer, resolved_query):
                issues.append("The code example is not complete or well-formed enough to be useful.")
            issues.extend(self.kb._code_request_topic_issues(answer, resolved_query))

        if recent_answers and self.kb._answer_too_similar_to_recent(answer, recent_answers):
            issues.append("The answer is too similar to a recent answer in the same chat.")

        return issues

    def _repair_answer(
        self,
        *,
        raw_query: str,
        resolved_query: str,
        history_text: str,
        session_summary: str,
        active_topic: str,
        task_profile: Dict[str, Any],
        coverage_level: str,
        evidence_text: str,
        related_kps: Sequence[Dict[str, Any]],
        allowed_source_ids: Sequence[str],
        current_payload: Dict[str, Any],
        issues: Sequence[str],
        repeat_instruction: str,
    ) -> Dict[str, Any]:
        allowed_text = ", ".join(allowed_source_ids) or "None"
        related_text = "\n".join(
            f"- {clean_display_text(item.get('name', ''))}"
            for item in related_kps
            if clean_display_text(item.get("name", ""))
        ) or "- None"
        issues_text = "\n".join(f"- {issue}" for issue in issues) or "- None"

        system_prompt = (
            "You repair one student-facing answer for a computer vision learning product. "
            "Return strict JSON only with exactly two keys: answer and used_sources."
        )
        user_prompt = textwrap.dedent(
            f"""
            Repair the current answer while keeping the supported content.

            Student message:
            {raw_query}

            Standalone question:
            {resolved_query}

            Recent conversation:
            {history_text}

            Session memory:
            - session_summary: {session_summary or 'None'}
            - active_topic: {active_topic or 'None'}

            Task guidance:
            {self.kb._task_prompt_guidance(task_profile)}

            Coverage level:
            {coverage_level}

            Relevant course topics:
            {related_text}

            Allowed source ids:
            {allowed_text}

            Evidence:
            {evidence_text}

            Current JSON:
            {json.dumps(current_payload, ensure_ascii=False)}

            Issues to fix:
            {issues_text}

            Variation requirement:
            {repeat_instruction}

            Repair rules:
            - Answer the standalone question directly.
            - Keep the wording natural and student-facing.
            - Do not mention source ids, file names, page numbers, retrieval, or system behavior.
            - If the topic is computer-vision related but the course evidence is partial, keep the conceptually correct explanation and only keep genuinely supporting used_sources.
            - If this is a code-only request, answer must be exactly one fenced Markdown code block and nothing else.
            - If this is a code-plus-explanation request, give a short explanation and then one fenced Markdown code block.
            - If more than one concept is asked, cover all of them explicitly.

            Return strict JSON only.
            """
        ).strip()

        raw_content = self.kb._llm_json_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.24,
        )
        payload = self.kb._parse_llm_answer_payload(
            raw_content,
            query=resolved_query,
            task_profile=task_profile,
            allowed_source_ids=allowed_source_ids,
        )
        return self._normalize_payload(
            payload,
            query=raw_query,
            task_profile=task_profile,
            related_kps=related_kps,
            citation_hits=[],
        )

    def _llm_answer(
        self,
        *,
        raw_query: str,
        resolved_query: str,
        history: Sequence[Dict[str, str]],
        resolution_context: Dict[str, Any],
        retrieval: Dict[str, Any],
    ) -> Dict[str, Any]:
        task_profile = retrieval["task_profile"]
        citations = retrieval["citations"]
        citation_hits = retrieval["citation_hits"]
        related_kps = retrieval["related_kps"]
        coverage_level = retrieval["coverage_level"]
        allowed_source_ids = self._allowed_source_ids(citations)

        history_text = self._history_text(history)
        evidence_text = self._evidence_text(citation_hits, citations)
        related_text = "\n".join(
            f"- {clean_display_text(item.get('name', ''))}"
            for item in related_kps
            if clean_display_text(item.get("name", ""))
        ) or "- None"
        target_concepts_text = "\n".join(
            f"- {clean_display_text(item.get('label', ''))}"
            for item in (task_profile.get("target_concepts") or [])
            if clean_display_text(item.get("label", ""))
        ) or "- None"
        repeat_context = self.kb._repeat_answer_context(resolved_query, history)
        repeat_instruction = (
            "This question has appeared before in the same chat. Keep the facts consistent, but vary the opening sentence and explanation path. "
            f"Preferred variation angle: {clean_display_text(repeat_context.get('style_instruction', ''))}"
            if repeat_context.get("repeat_count")
            else "No repeat-answer constraint beyond writing naturally."
        )

        system_prompt = textwrap.dedent(
            """
            You are the student-facing chat assistant for Course 4186.
            First answer the student's actual question well, then package the reply as JSON only.
            Return strict JSON with exactly two keys:
            - answer
            - used_sources

            Answering policy:
            - Use the standalone resolved question as the primary task.
            - Use recent conversation and session memory only to preserve continuity.
            - If the topic is clearly within computer vision, answer it even when lecture evidence is partial or missing.
            - Use course evidence only when it genuinely supports the final answer.
            - Never mention source ids, retrieval, system behavior, file names, or page numbers inside the answer body.
            - Keep the tone natural, polished, and student-facing rather than robotic or demo-like.
            """
        ).strip()
        user_prompt = textwrap.dedent(
            f"""
            Student message:
            {raw_query}

            Standalone resolved question:
            {resolved_query}

            Recent conversation:
            {history_text}

            Session memory:
            - session_summary: {clean_display_text(resolution_context.get('session_summary', '')) or 'None'}
            - active_topic: {clean_display_text(resolution_context.get('active_topic', '')) or 'None'}
            - followup_detected: {str(bool(resolution_context.get('followup_detected'))).lower()}
            - used_history: {str(bool(resolution_context.get('used_history'))).lower()}
            - focus_concept: {clean_display_text(resolution_context.get('focus_concept', '')) or 'None'}
            - anchor_query: {clean_display_text(resolution_context.get('anchor_query', '')) or 'None'}

            Task guidance:
            {self.kb._task_prompt_guidance(task_profile)}

            Coverage level:
            {coverage_level}

            Related course topics:
            {related_text}

            Target concepts to cover:
            {target_concepts_text}

            Allowed source ids:
            {", ".join(allowed_source_ids) or 'None'}

            Evidence:
            {evidence_text}

            Variation requirement:
            {repeat_instruction}

            Output rules:
            - Answer the standalone question directly and completely.
            - If the student asked for code only, put exactly one fenced Markdown code block in answer and nothing else.
            - If the student asked for both explanation and code, give a short explanation and then one fenced Markdown code block.
            - If the student asked about multiple concepts, address all of them explicitly.
            - If coverage level is direct and this is not a code-only request, keep at least one genuinely supporting source id.
            - If reliable lecture support exists, select only the source ids that truly support the final answer.
            - If lecture support is weak or absent but the topic is still computer vision, answer from reliable domain knowledge and leave used_sources empty or partial as appropriate.
            - Do not include markdown headings unless the question clearly needs them.
            - Do not output anything outside the JSON object.
            """
        ).strip()

        raw_content = self.kb._llm_json_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.10 if task_profile.get("is_code_request") else (0.28 if repeat_context.get("repeat_count") else 0.18),
        )
        payload = self.kb._parse_llm_answer_payload(
            raw_content,
            query=resolved_query,
            task_profile=task_profile,
            allowed_source_ids=allowed_source_ids,
        )
        payload = self._normalize_payload(
            payload,
            query=raw_query,
            task_profile=task_profile,
            related_kps=related_kps,
            citation_hits=citation_hits,
        )
        issues = self._answer_issues(
            payload,
            resolved_query=resolved_query,
            task_profile=task_profile,
            allowed_source_ids=allowed_source_ids,
            coverage_level=coverage_level,
            citations=citations,
            recent_answers=list(repeat_context.get("recent_answers") or []),
        )
        repair_rounds = 0
        while issues and repair_rounds < 2:
            payload = self._repair_answer(
                raw_query=raw_query,
                resolved_query=resolved_query,
                history_text=history_text,
                session_summary=clean_display_text(resolution_context.get("session_summary", "")),
                active_topic=clean_display_text(resolution_context.get("active_topic", "")),
                task_profile=task_profile,
                coverage_level=coverage_level,
                evidence_text=evidence_text,
                related_kps=related_kps,
                allowed_source_ids=allowed_source_ids,
                current_payload=payload,
                issues=issues,
                repeat_instruction=repeat_instruction,
            )
            payload = self._normalize_payload(
                payload,
                query=raw_query,
                task_profile=task_profile,
                related_kps=related_kps,
                citation_hits=citation_hits,
            )
            issues = self._answer_issues(
                payload,
                resolved_query=resolved_query,
                task_profile=task_profile,
                allowed_source_ids=allowed_source_ids,
                coverage_level=coverage_level,
                citations=citations,
                recent_answers=list(repeat_context.get("recent_answers") or []),
            )
            repair_rounds += 1
        if issues:
            raise ValueError("Chat pipeline answer did not pass validation: " + " | ".join(issues))
        return payload

    def _code_fallback_payload(
        self,
        *,
        raw_query: str,
        resolved_query: str,
        history: Sequence[Dict[str, str]],
        resolution_context: Dict[str, Any],
        retrieval: Dict[str, Any],
    ) -> Dict[str, Any]:
        task_profile = retrieval["task_profile"]
        if self.kb._llm_client is None or not self.kb._llm_model:
            if task_profile.get("is_code_only_request"):
                return {
                    "answer": "I could not generate a reliable code example right now. Please try again or ask for a smaller code task.",
                    "used_sources": [],
                }
            return {
                "answer": (
                    "I could explain the concept, but I could not generate a reliable code example right now. "
                    "Please try again or ask for a smaller coding subtask."
                ),
                "used_sources": [],
            }

        citations = retrieval["citations"]
        citation_hits = retrieval["citation_hits"]
        related_kps = retrieval["related_kps"]
        allowed_source_ids = self._allowed_source_ids(citations)
        prefs = dict(task_profile.get("request_preferences") or {})
        language = clean_display_text(prefs.get("language", "")) or "the requested language"
        framework = clean_display_text(prefs.get("framework", "")) or "None"
        history_text = self._history_text(history)
        evidence_text = self._evidence_text(citation_hits, citations)
        related_text = "\n".join(
            f"- {clean_display_text(item.get('name', ''))}"
            for item in related_kps
            if clean_display_text(item.get("name", ""))
        ) or "- None"

        system_prompt = textwrap.dedent(
            """
            You generate code-focused answers for a commercial student-facing computer vision assistant.
            Return strict JSON only with exactly two keys:
            - answer
            - used_sources

            Requirements:
            - The answer must be directly usable by a student.
            - Prefer a small, honest, runnable example over pseudo-code placeholders.
            - Never leak source ids, retrieval details, file names, or page numbers into the answer body.
            """
        ).strip()
        user_prompt = textwrap.dedent(
            f"""
            Student message:
            {raw_query}

            Standalone resolved question:
            {resolved_query}

            Recent conversation:
            {history_text}

            Session memory:
            - session_summary: {clean_display_text(resolution_context.get('session_summary', '')) or 'None'}
            - active_topic: {clean_display_text(resolution_context.get('active_topic', '')) or 'None'}

            Request preferences:
            - language: {language}
            - framework: {framework}
            - code_request_mode: {clean_display_text(task_profile.get('code_request_mode', '')) or 'code_plus_explanation'}

            Related course topics:
            {related_text}

            Evidence:
            {evidence_text}

            Allowed source ids:
            {", ".join(allowed_source_ids) or 'None'}

            Code-generation rules:
            - Focus on delivering the code the student asked for.
            - If code_request_mode is code_only, answer with exactly one fenced Markdown code block and nothing else.
            - If code_request_mode is code_plus_explanation, first give a short explanation, then one fenced Markdown code block.
            - If the question asks for PyTorch convolution, use a 2D example with valid tensor shapes.
            - If the question asks for epipolar geometry code, prefer a small honest example of the epipolar constraint, fundamental matrix usage, or epiline computation. Do not invent a fake full pipeline with placeholder variables.
            - If course evidence directly supports the concept explanation, keep only genuinely supporting used_sources.
            - If the code itself goes beyond the lecture details, keep the concept explanation grounded and the code technically correct.

            Return strict JSON only.
            """
        ).strip()

        raw_content = self.kb._llm_json_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        payload = self.kb._parse_llm_answer_payload(
            raw_content,
            query=resolved_query,
            task_profile=task_profile,
            allowed_source_ids=allowed_source_ids,
        )
        payload = self._normalize_payload(
            payload,
            query=raw_query,
            task_profile=task_profile,
            related_kps=related_kps,
            citation_hits=citation_hits,
        )
        issues = self._answer_issues(
            payload,
            resolved_query=resolved_query,
            task_profile=task_profile,
            allowed_source_ids=allowed_source_ids,
            coverage_level=retrieval["coverage_level"],
            citations=citations,
            recent_answers=[],
        )
        if issues:
            raise ValueError("Code fallback answer did not pass validation: " + " | ".join(issues))
        return payload

    def _finalize_response(
        self,
        *,
        raw_query: str,
        history: Sequence[Dict[str, str]],
        resolved_query: str,
        resolution_context: Dict[str, Any],
        retrieval: Dict[str, Any],
        payload: Dict[str, Any],
        previous_memory: Optional[Dict[str, Any]],
        mode: str,
    ) -> Dict[str, Any]:
        task_profile = retrieval["task_profile"]
        citations = list(retrieval["citations"] or [])
        used_sources = [
            clean_display_text(item).upper()
            for item in payload.get("used_sources", [])
            if clean_display_text(item)
        ]

        if task_profile.get("is_code_only_request"):
            final_citations: List[Dict[str, Any]] = []
        elif used_sources:
            final_citations = self.kb._select_final_citations(citations, used_sources, max_count=3)
        else:
            final_citations = []

        answer_body = str(payload.get("answer") or "").strip()
        if not final_citations:
            answer_body = self.kb._strip_uncited_course_connection(answer_body)

        answer_text = self.kb._answer_with_source_line(answer_body, final_citations)
        session_memory = self.kb.build_session_memory(
            raw_query,
            answer_text,
            history,
            previous_memory=previous_memory,
            mode=mode,
            resolved_query=resolved_query,
            resolution_context=resolution_context,
            related=retrieval["related_kps"],
        )
        return {
            "answer": answer_text,
            "citations": final_citations,
            "related_kps": list(retrieval["related_kps"] or []),
            "mode": mode,
            "session_memory": session_memory,
            "_resolved_query": resolved_query,
            "_resolution_context": dict(resolution_context or {}),
        }

    def _fallback_payload(
        self,
        *,
        resolved_query: str,
        retrieval: Dict[str, Any],
    ) -> Dict[str, Any]:
        task_profile = retrieval["task_profile"]
        citations = list(retrieval["citations"] or [])
        related_kps = list(retrieval["related_kps"] or [])
        citation_hits = list(retrieval["citation_hits"] or [])

        if task_profile.get("is_code_request"):
            if task_profile.get("is_code_only_request"):
                return {
                    "answer": "I could not generate a reliable code example right now. Please try again or ask for a smaller code task.",
                    "used_sources": [],
                }
            return {
                "answer": (
                    "I could explain the concept, but I could not generate a reliable code example right now. "
                    "Please try again or ask for a smaller coding subtask."
                ),
                "used_sources": [],
            }

        short_answer = ""
        if citations and self.kb._looks_brief_concept_request(
            resolved_query,
            task_profile=task_profile,
            kp_hits=retrieval["kp_hits"],
        ):
            short_answer = self.kb._grounded_short_answer(resolved_query, citation_hits[:3], citations[:3]) or ""

        answer_text = short_answer or self.kb._fallback_answer(
            resolved_query,
            related_kps,
            citation_hits[:4],
            citations,
            task_profile=task_profile,
        )
        return {
            "answer": answer_text,
            "used_sources": [
                clean_display_text(item.get("citation_id", "")).upper()
                for item in citations[:2]
                if clean_display_text(item.get("citation_id", ""))
            ],
        }

    def answer(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5,
        session_memory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        history = history or []
        raw_query = clean_display_text(query)
        resolution_context = self.kb._resolve_query_with_context(
            raw_query,
            history,
            session_memory=session_memory,
        )
        resolved_query = clean_display_text(resolution_context.get("resolved_query", "")) or raw_query

        special_response = self.kb._assistant_response_for_query(raw_query, history)
        if special_response:
            session_memory_update = self.kb.build_session_memory(
                raw_query,
                special_response,
                history,
                previous_memory=session_memory,
                mode="assistant",
                resolved_query=resolved_query,
                resolution_context=resolution_context,
                related=[],
            )
            return {
                "answer": special_response,
                "citations": [],
                "related_kps": [],
                "mode": "assistant",
                "session_memory": session_memory_update,
                "_resolved_query": resolved_query,
                "_resolution_context": dict(resolution_context or {}),
            }

        in_scope = self.kb._looks_cv_request(resolved_query) or self.kb._looks_course_request(resolved_query)
        if not in_scope:
            answer_text = self.kb._out_of_scope_response(raw_query, history)
            session_memory_update = self.kb.build_session_memory(
                raw_query,
                answer_text,
                history,
                previous_memory=session_memory,
                mode="assistant",
                resolved_query=resolved_query,
                resolution_context=resolution_context,
                related=[],
            )
            return {
                "answer": answer_text,
                "citations": [],
                "related_kps": [],
                "mode": "assistant",
                "session_memory": session_memory_update,
                "_resolved_query": resolved_query,
                "_resolution_context": dict(resolution_context or {}),
            }

        retrieval = self._retrieve_context(raw_query, resolved_query, top_k=top_k)
        task_profile = retrieval["task_profile"]
        if task_profile.get("is_question_generation_request"):
            practice_response = self.kb._practice_question_response(
                raw_query,
                history,
                retrieval["related_kps"],
                retrieval["citation_hits"][:4],
                retrieval["citations"],
                task_profile=task_profile,
                resolved_query=resolved_query,
                resolution_context=resolution_context,
            )
            payload = {
                "answer": str(practice_response.get("answer") or "").strip(),
                "used_sources": [
                    clean_display_text(item.get("citation_id", "")).upper()
                    for item in practice_response.get("citations", [])
                    if clean_display_text(item.get("citation_id", ""))
                ],
            }
            patched_retrieval = dict(retrieval)
            patched_retrieval["citations"] = list(practice_response.get("citations", []))
            return self._finalize_response(
                raw_query=raw_query,
                history=history,
                resolved_query=resolved_query,
                resolution_context=resolution_context,
                retrieval=patched_retrieval,
                payload=payload,
                previous_memory=session_memory,
                mode=practice_response.get("mode", "practice_question"),
            )

        llm_available = self.kb._llm_client is not None and bool(self.kb._llm_model)
        if llm_available and task_profile.get("is_code_request"):
            try:
                payload = self._code_fallback_payload(
                    raw_query=raw_query,
                    resolved_query=resolved_query,
                    history=history,
                    resolution_context=resolution_context,
                    retrieval=retrieval,
                )
                return self._finalize_response(
                    raw_query=raw_query,
                    history=history,
                    resolved_query=resolved_query,
                    resolution_context=resolution_context,
                    retrieval=retrieval,
                    payload=payload,
                    previous_memory=session_memory,
                    mode="code_llm_v2",
                )
            except Exception:
                pass

        if llm_available:
            try:
                payload = self._llm_answer(
                    raw_query=raw_query,
                    resolved_query=resolved_query,
                    history=history,
                    resolution_context=resolution_context,
                    retrieval=retrieval,
                )
                return self._finalize_response(
                    raw_query=raw_query,
                    history=history,
                    resolved_query=resolved_query,
                    resolution_context=resolution_context,
                    retrieval=retrieval,
                    payload=payload,
                    previous_memory=session_memory,
                    mode="llm_v2",
                )
            except Exception:
                if task_profile.get("is_code_request"):
                    try:
                        payload = self._code_fallback_payload(
                            raw_query=raw_query,
                            resolved_query=resolved_query,
                            history=history,
                            resolution_context=resolution_context,
                            retrieval=retrieval,
                        )
                        return self._finalize_response(
                            raw_query=raw_query,
                            history=history,
                            resolved_query=resolved_query,
                            resolution_context=resolution_context,
                            retrieval=retrieval,
                            payload=payload,
                            previous_memory=session_memory,
                            mode="code_fallback_v2",
                        )
                    except Exception:
                        pass

        fallback_payload = self._fallback_payload(
            resolved_query=resolved_query,
            retrieval=retrieval,
        )
        return self._finalize_response(
            raw_query=raw_query,
            history=history,
            resolved_query=resolved_query,
            resolution_context=resolution_context,
            retrieval=retrieval,
            payload=fallback_payload,
            previous_memory=session_memory,
            mode="fallback_v2",
        )
