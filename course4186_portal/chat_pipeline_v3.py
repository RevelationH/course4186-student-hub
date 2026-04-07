from __future__ import annotations

import json
import re
import textwrap
from typing import Any, Dict, List, Optional, Sequence

from course4186_portal.chat_code_templates import build_code_template
from course4186_portal.chat_pipeline_v2 import Course4186ChatPipeline as BaseCourse4186ChatPipeline
from course4186_portal.kb_service import clean_display_text


class Course4186ChatPipeline(BaseCourse4186ChatPipeline):
    """Unified chat pipeline for Course 4186.

    The design goal here is to keep one stable answer chain:
    1. resolve the user query with chat context
    2. determine scope once
    3. retrieve supporting material once
    4. generate one student-facing answer
    5. finalize citations and session memory consistently

    We still keep a single extractive fallback if the LLM path fails, but
    code requests, follow-ups, and partially grounded CV questions all go
    through the same generation path instead of separate branches.
    """

    NON_COURSE_SIGNAL_PATTERNS = (
        re.compile(r"\b(weather|restaurant|restaurants|bitcoin|crypto|stock|stocks|economics|economic|finance|financial)\b", re.IGNORECASE),
        re.compile(r"\b(president|prime minister|election|government|politics|political)\b", re.IGNORECASE),
        re.compile(r"\b(joke|funny|married|wife|husband|boyfriend|girlfriend|dating)\b", re.IGNORECASE),
        re.compile(r"\b(driving test|driver'?s test|restaurant|travel|flight|hotel)\b", re.IGNORECASE),
        re.compile(r"\b(job application|job applications|resume|curriculum vitae|cover letter|linkedin)\b", re.IGNORECASE),
    )
    CV_RESUME_QUERY_RE = re.compile(r"\bcv\b", re.IGNORECASE)
    REFERENCE_REQUEST_RE = re.compile(
        r"\b("
        r"show me the source|show the source|which source|which sources|"
        r"which pdf|which lecture|which lectures|which page|"
        r"where is this explained|where in the course do we discuss|where do we discuss|"
        r"point me to the lecture|"
        r"which lecture should i review|which lecture should i revise|"
        r"which pdf should i open|which material should i open|"
        r"what should i review|what should i revise|"
        r"relevant source|open first|review for that|revise for that"
        r")\b",
        re.IGNORECASE,
    )
    REFERENCE_OPEN_RE = re.compile(r"\b(open|pdf|source|page)\b", re.IGNORECASE)
    REFERENCE_REVIEW_RE = re.compile(r"\b(review|revise|lecture|lectures)\b", re.IGNORECASE)
    GENERIC_REFERENCE_LABELS = {
        "lecture", "lectures", "source", "sources", "pdf", "page", "pages",
        "review", "revise", "open", "material", "materials", "topic", "topics",
        "section", "sections", "course", "first", "relevant", "related",
        "covered", "cover", "directly", "explained",
    }
    REFERENCE_TOPIC_DISPLAY_ALIASES = {
        "sift": "SIFT",
        "sfm": "SfM",
        "cnn": "CNN",
        "nerf": "NeRF",
        "log": "LoG",
    }
    OUT_OF_SCOPE_REFUSAL_MARKERS = (
        "outside course 4186",
        "outside the scope of course 4186",
        "won't answer it directly",
        "won’t answer it directly",
        "not answer it directly",
        "not answer this directly",
    )

    def _asks_course_coverage(self, query: str) -> bool:
        lowered = clean_display_text(query).lower()
        markers = (
            "covered in this course",
            "covered in the course",
            "covered in this lecture",
            "covered in the lecture",
            "covered in the lectures",
            "in this course",
            "in the lecture",
            "in the lectures",
        )
        return any(marker in lowered for marker in markers)

    def _assistant_response_payload(
        self,
        *,
        answer_text: str,
        raw_query: str,
        history: Sequence[Dict[str, str]],
        previous_memory: Optional[Dict[str, Any]],
        resolved_query: str,
        resolution_context: Optional[Dict[str, Any]] = None,
        mode: str = "assistant",
    ) -> Dict[str, Any]:
        session_memory = self.kb.build_session_memory(
            raw_query,
            answer_text,
            history,
            previous_memory=previous_memory,
            mode=mode,
            resolved_query=resolved_query,
            resolution_context=resolution_context,
            related=[],
        )
        return {
            "answer": answer_text,
            "citations": [],
            "related_kps": [],
            "mode": mode,
            "session_memory": session_memory,
            "_resolved_query": resolved_query,
            "_resolution_context": dict(resolution_context or {}),
        }

    def _strict_out_of_scope_response(
        self,
        raw_query: str,
        history: Sequence[Dict[str, str]],
    ) -> str:
        candidate = clean_display_text(self.kb._out_of_scope_response(raw_query, history))
        lowered = candidate.lower()
        if any(marker in lowered for marker in self.OUT_OF_SCOPE_REFUSAL_MARKERS):
            return candidate

        variant_index = (sum(ord(ch) for ch in clean_display_text(raw_query)) + len(history)) % 3
        refusals = [
            "That question is outside Course 4186, so I won't answer it directly.",
            "I focus on Course 4186, so I won't answer that question directly.",
            "That request falls outside Course 4186, so I should not answer it directly.",
        ]
        redirects = [
            "If you want, ask about topics such as convolution, SIFT, homography, stereo vision, or optical flow.",
            "If you want to stay within the course, I can help with topics like image filtering, feature matching, epipolar geometry, or object detection.",
            "If you want to keep the discussion course-related, I can help with camera geometry, SFM, optical flow, or quiz preparation.",
        ]
        return f"{refusals[variant_index]} {redirects[variant_index]}"

    def _raw_cv_signal_score(self, query: str) -> int:
        lowered = clean_display_text(query).lower()
        if not lowered:
            return 0
        phrase_hits = (
            sum(1 for phrase in (
                "computer vision",
                "object detection",
                "image segmentation",
                "optical flow",
                "epipolar geometry",
                "structure from motion",
                "stereo vision",
                "camera calibration",
                "image filtering",
                "neural radiance field",
            ) if phrase in lowered)
        )
        tokens = {
            token
            for token in re.findall(r"[A-Za-z][A-Za-z0-9+-]*", lowered)
        }
        token_hits = len(tokens & {
            "image", "images", "video", "camera", "convolution", "filtering", "kernel",
            "edge", "corner", "feature", "descriptor", "matching", "segmentation",
            "object", "detection", "recognition", "tracking", "geometry", "epipolar",
            "stereo", "disparity", "depth", "optical", "flow", "homography", "calibration",
            "sfm", "sift", "harris", "nerf", "radiance", "cnn", "vision", "reconstruction",
        })
        return phrase_hits * 3 + token_hits

    def _non_course_signal_score(self, query: str) -> int:
        lowered = clean_display_text(query).lower()
        if not lowered:
            return 0
        score = 0
        for pattern in self.NON_COURSE_SIGNAL_PATTERNS:
            if pattern.search(lowered):
                score += 2
        if self.CV_RESUME_QUERY_RE.search(lowered) and re.search(
            r"\b(write|improve|submit|job|jobs|application|applications|resume|curriculum vitae|cover letter|linkedin)\b",
            lowered,
            re.IGNORECASE,
        ):
            score += 3
        return score

    def _asks_for_reference(self, query: str) -> bool:
        lowered = clean_display_text(query).lower()
        if not lowered:
            return False
        return bool(self.REFERENCE_REQUEST_RE.search(lowered))

    def _reference_request_kind(self, query: str) -> str:
        lowered = clean_display_text(query).lower()
        if not lowered:
            return "source"
        if self.REFERENCE_OPEN_RE.search(lowered):
            return "open"
        if self.REFERENCE_REVIEW_RE.search(lowered):
            return "review"
        return "source"

    def _clean_reference_label(self, value: str) -> str:
        label = clean_display_text(value)
        if not label:
            return ""
        label = re.sub(r"(?i)^source:\s*", "", label).strip()
        label = re.sub(r"(?i)\baka\.?\s*$", "", label).strip(" -:;,.")
        label = re.sub(r"\bimageswith\b", "images with", label, flags=re.IGNORECASE)
        label = re.sub(r"\s{2,}", " ", label).strip()
        lowered = label.lower()
        if not label or lowered in {"source", "the lectures", "lecture materials"}:
            return ""
        if "source:" in lowered:
            return ""
        if re.fullmatch(r"[A-Z](?:\.[A-Z]?)*\s+[A-Z][a-z]+", label):
            return ""
        if len(label) < 4:
            return ""
        return label

    def _is_generic_reference_label(self, value: str) -> bool:
        label = self._clean_reference_label(value)
        if not label:
            return True
        lowered = label.lower()
        if lowered in self.GENERIC_REFERENCE_LABELS:
            return True
        tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9+-]*", lowered)
        ]
        if not tokens:
            return True
        meaningful_tokens = [
            token
            for token in tokens
            if token not in self.GENERIC_REFERENCE_LABELS
        ]
        return len(meaningful_tokens) == 0

    def _format_reference_topic(self, value: str) -> str:
        label = self._clean_reference_label(value)
        if not label or self._is_generic_reference_label(label):
            return ""

        def repl(match: re.Match[str]) -> str:
            token = match.group(0)
            return self.REFERENCE_TOPIC_DISPLAY_ALIASES.get(token.lower(), token)

        formatted = re.sub(r"[A-Za-z][A-Za-z0-9+-]*", repl, label)
        return clean_display_text(formatted)

    def _best_related_topic_label(
        self,
        raw_query: str,
        retrieval: Dict[str, Any],
    ) -> str:
        task_profile = dict(retrieval.get("task_profile") or {})
        query_terms = {
            token
            for token in re.findall(r"[A-Za-z][A-Za-z0-9+-]*", clean_display_text(raw_query).lower())
            if len(token) > 2
        }
        for concept in task_profile.get("target_concepts") or []:
            query_terms.update(
                clean_display_text(token).lower()
                for token in concept.get("tokens", set())
                if clean_display_text(token)
            )

        for item in retrieval.get("related_kps") or []:
            label = self._format_reference_topic(clean_display_text(item.get("name", "")))
            if not label:
                continue
            label_terms = {
                token
                for token in re.findall(r"[A-Za-z][A-Za-z0-9+-]*", label.lower())
                if len(token) > 2
            }
            if query_terms and not (label_terms & query_terms):
                continue
            return label
        return ""

    def _reference_primary_topic(
        self,
        raw_query: str,
        retrieval: Dict[str, Any],
    ) -> str:
        task_profile = dict(retrieval.get("task_profile") or {})
        for concept in task_profile.get("target_concepts") or []:
            label = self._format_reference_topic(clean_display_text(concept.get("label", "")))
            if label:
                return label

        related_label = self._best_related_topic_label(raw_query, retrieval)
        if related_label:
            return related_label

        citations = list(retrieval.get("citations") or [])
        if citations:
            return self._citation_topic_label(citations[0])
        return "the most relevant topic"

    def _citation_topic_label(
        self,
        citation: Dict[str, Any],
    ) -> str:
        section = self._format_reference_topic(clean_display_text(citation.get("section", "")))
        if section and section.lower() not in {"untitled section", "source", "page"} and len(section) >= 4:
            return section
        return "the most relevant lecture section"

    def _reference_guidance_payload(
        self,
        *,
        raw_query: str,
        retrieval: Dict[str, Any],
    ) -> Dict[str, Any]:
        citations = list(retrieval.get("citations") or [])
        related_kps = list(retrieval.get("related_kps") or [])
        if not citations:
            return {
                "answer": (
                    "I could not find a strong lecture match for that request yet. "
                    "Try naming the topic more specifically, and then use the source buttons once the lecture match appears."
                ),
                "used_sources": [],
        }

        kind = self._reference_request_kind(raw_query)
        primary = self._reference_primary_topic(raw_query, retrieval)

        if kind == "open":
            opener = f"Start with the lecture section on {primary}."
        elif kind == "review":
            opener = f"Start by reviewing the lecture section on {primary}."
        else:
            opener = f"The most relevant place to start is the lecture section on {primary}."

        follow_up = ""
        if len(citations) > 1:
            follow_up = " Then use the second source if you want a worked example or a more applied explanation."

        closing = " Use the source buttons below to open the exact PDF pages."
        return {
            "answer": opener + follow_up + closing,
            "used_sources": [
                clean_display_text(item.get("citation_id", "")).upper()
                for item in citations[:2]
                if clean_display_text(item.get("citation_id", ""))
            ],
        }

    def _scope_candidates(
        self,
        raw_query: str,
        resolved_query: str,
        resolution_context: Dict[str, Any],
    ) -> List[str]:
        rows: List[str] = []
        focus_concept = clean_display_text(resolution_context.get("focus_concept", ""))
        anchor_query = clean_display_text(resolution_context.get("anchor_query", ""))
        followup_detected = bool(resolution_context.get("followup_detected"))

        for candidate in (
            resolved_query,
            raw_query,
            f"{resolved_query} about {focus_concept}" if followup_detected and focus_concept else "",
            f"{resolved_query} Context: {anchor_query}" if followup_detected and anchor_query else "",
        ):
            cleaned = clean_display_text(candidate)
            if cleaned and cleaned not in rows:
                rows.append(cleaned)
        return rows

    def _is_in_scope(
        self,
        raw_query: str,
        resolved_query: str,
        resolution_context: Dict[str, Any],
    ) -> bool:
        negative_signal = max(
            self._non_course_signal_score(raw_query),
            self._non_course_signal_score(resolved_query),
        )
        positive_signal = max(
            self._raw_cv_signal_score(raw_query),
            self._raw_cv_signal_score(resolved_query),
        )

        if negative_signal >= 2 and positive_signal == 0:
            return False
        if negative_signal >= 3 and positive_signal < 2:
            return False

        for candidate in self._scope_candidates(raw_query, resolved_query, resolution_context):
            if self.kb._looks_course_request(candidate) or self.kb._looks_cv_request(candidate):
                return True
        return False

    def _effective_retrieval_query(
        self,
        raw_query: str,
        resolved_query: str,
        resolution_context: Dict[str, Any],
        history: Sequence[Dict[str, str]],
        session_memory: Optional[Dict[str, Any]],
    ) -> str:
        resolved = clean_display_text(resolved_query) or clean_display_text(raw_query)
        followup_detected = bool(resolution_context.get("followup_detected"))
        current_labels = self.kb._query_concept_labels(raw_query, limit=2)
        strong_focus = self.kb._query_has_strong_focus(raw_query)
        session_focus = self.kb._session_context_focus(history, session_memory)
        focus_concept = (
            clean_display_text(session_focus.get("focus_concept", ""))
            or clean_display_text(resolution_context.get("focus_concept", ""))
            or clean_display_text(session_focus.get("active_topic", ""))
        )

        if focus_concept and focus_concept.lower() not in resolved.lower():
            if followup_detected:
                return clean_display_text(f"{resolved} about {focus_concept}")
            if not strong_focus and not current_labels:
                return clean_display_text(f"{resolved} about {focus_concept}")
        return resolved

    def _refine_retrieval(self, retrieval: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(retrieval or {})
        task_profile = dict(normalized.get("task_profile") or {})
        concepts = list(task_profile.get("target_concepts") or [])
        if not concepts:
            normalized["task_profile"] = task_profile
            return normalized

        candidate_hits = list(normalized.get("citation_hits") or []) or list(normalized.get("chunk_hits") or [])
        matched_concepts: List[Dict[str, Any]] = []
        for concept in concepts:
            if any(self.kb._concept_matches_chunk_item(concept, hit.get("item") or {}) for hit in candidate_hits[:8]):
                matched_concepts.append(concept)

        if matched_concepts:
            task_profile["target_concepts"] = matched_concepts[:3]
            if task_profile.get("is_code_request") and len(matched_concepts) <= 1:
                task_profile["needs_multi_concept_coverage"] = False
            normalized["task_profile"] = task_profile
            return normalized

        if task_profile.get("is_code_request"):
            relevant_kp_hits = list(task_profile.get("relevant_kp_hits") or normalized.get("kp_hits") or [])
            kp_concepts: List[Dict[str, Any]] = []
            seen_labels: set[str] = set()
            for index, hit in enumerate(relevant_kp_hits[:2]):
                item = dict(hit.get("item") or {})
                label = clean_display_text(item.get("name", ""))
                lowered = label.lower()
                if not label or lowered in seen_labels:
                    continue
                seen_labels.add(lowered)
                kp_concepts.append(
                    {
                        "label": label,
                        "phrases": {lowered},
                        "tokens": set(self.kb._concept_tokens(label)),
                        "position": index,
                        "source": "kp",
                    }
                )
            if kp_concepts:
                task_profile["target_concepts"] = kp_concepts
                task_profile["needs_multi_concept_coverage"] = len(kp_concepts) > 1

        normalized["task_profile"] = task_profile
        return normalized

    def _repeat_instruction(self, resolved_query: str, history: Sequence[Dict[str, str]]) -> str:
        repeat_context = self.kb._repeat_answer_context(resolved_query, history)
        if repeat_context.get("repeat_count"):
            style = clean_display_text(repeat_context.get("style_instruction", "")) or "Use a different explanation path."
            return (
                "This question has appeared before in the same chat. Keep the facts consistent, "
                f"but vary the wording and explanation path. Preferred variation angle: {style}"
            )
        return "Write naturally. There is no special repeat-answer constraint."

    def _exact_query_repeat_count(
        self,
        query: str,
        history: Sequence[Dict[str, str]],
    ) -> int:
        cleaned = clean_display_text(query).lower()
        if not cleaned:
            return 0
        count = 0
        for turn in history:
            if str(turn.get("role") or "").lower() != "user":
                continue
            if clean_display_text(turn.get("content", "")).lower() == cleaned:
                count += 1
        return count

    def _autofill_used_sources(
        self,
        payload: Dict[str, Any],
        *,
        retrieval: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized = dict(payload or {})
        citations = list(retrieval.get("citations") or [])
        citation_hits = list(retrieval.get("citation_hits") or [])
        task_profile = dict(retrieval.get("task_profile") or {})
        coverage_level = clean_display_text(retrieval.get("coverage_level", "")) or "none"

        allowed_source_ids = {
            clean_display_text(item.get("citation_id", "")).upper()
            for item in citations
            if clean_display_text(item.get("citation_id", ""))
        }
        explicit_ids: List[str] = []
        for item in normalized.get("used_sources", []):
            source_id = clean_display_text(item).upper()
            if source_id and source_id in allowed_source_ids and source_id not in explicit_ids:
                explicit_ids.append(source_id)
        if explicit_ids:
            normalized["used_sources"] = explicit_ids
            return normalized

        if not citations:
            normalized["used_sources"] = []
            return normalized

        target_concepts = list(task_profile.get("target_concepts") or [])
        scored_rows: List[Dict[str, Any]] = []
        for index, citation in enumerate(citations):
            source_id = clean_display_text(citation.get("citation_id", "")).upper()
            if not source_id:
                continue
            hit_item = citation_hits[index].get("item", {}) if index < len(citation_hits) else {}
            concept_score = 0.0
            for concept in target_concepts[:3]:
                if self.kb._concept_matches_chunk_item(concept, hit_item):
                    concept_score += 2.0
            if not target_concepts:
                concept_score = max(concept_score, 1.5 - 0.25 * index)
            scored_rows.append(
                {
                    "source_id": source_id,
                    "score": concept_score,
                    "index": index,
                }
            )

        if not scored_rows:
            normalized["used_sources"] = []
            return normalized

        scored_rows.sort(key=lambda row: (-float(row["score"]), int(row["index"])))
        if target_concepts:
            desired_count = 1 if len(target_concepts) == 1 else min(3, len(target_concepts))
        else:
            desired_count = 1
        if task_profile.get("is_code_request"):
            desired_count = max(desired_count, 1)
        desired_count = min(desired_count, len(scored_rows))

        positive_rows = [row for row in scored_rows if float(row["score"]) > 0.0]
        if coverage_level == "none" and not positive_rows:
            normalized["used_sources"] = []
            return normalized
        source_rows = positive_rows[:desired_count] if positive_rows else scored_rows[:desired_count]
        normalized["used_sources"] = [row["source_id"] for row in source_rows]
        return normalized

    def _repair_unified_payload(
        self,
        *,
        raw_query: str,
        resolved_query: str,
        history_text: str,
        session_summary: str,
        active_topic: str,
        task_profile: Dict[str, Any],
        coverage_level: str,
        related_kps: Sequence[Dict[str, Any]],
        evidence_text: str,
        allowed_source_ids: Sequence[str],
        current_payload: Dict[str, Any],
        issues: Sequence[str],
        repeat_instruction: str,
    ) -> Dict[str, Any]:
        related_text = "\n".join(
            f"- {clean_display_text(item.get('name', ''))}"
            for item in related_kps
            if clean_display_text(item.get("name", ""))
        ) or "- None"
        issues_text = "\n".join(f"- {issue}" for issue in issues) or "- None"
        code_repair_rules = ""
        if task_profile.get("is_code_request"):
            code_repair_rules = textwrap.dedent(
                """
                Additional code-repair rules:
                - Return a short runnable example, not pseudo-code.
                - Use concrete numeric inputs or clearly defined variables.
                - Do not use ellipses.
                - Do not use placeholder identifiers like x1' or y1'.
                - If the request is code-only, return only one fenced Markdown code block.
                """
            ).strip()

        system_prompt = (
            "You repair one student-facing answer for a Course 4186 computer vision assistant. "
            "Return strict JSON only with exactly two keys: answer and used_sources."
        )
        user_prompt = textwrap.dedent(
            f"""
            Repair the current answer while keeping the supported content and preserving a natural student-facing tone.

            Student message:
            {raw_query}

            Standalone resolved question:
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
            {", ".join(allowed_source_ids) or 'None'}

            Evidence:
            {evidence_text}

            Current JSON:
            {json.dumps(current_payload, ensure_ascii=False)}

            Issues to fix:
            {issues_text}

            Variation requirement:
            {repeat_instruction}

            Repair rules:
            - Answer the actual standalone question directly.
            - Keep the answer natural, polished, and student-facing.
            - Do not mention retrieval, file names, page numbers, source ids, or system behavior.
            - If the student asked whether a topic is covered in the course, answer that explicitly using the coverage level and evidence.
            - If the topic is computer-vision related but the course evidence is partial or missing, keep the concept explanation correct and leave used_sources empty or partial as appropriate.
            - If this is a code-only request, answer must be exactly one fenced Markdown code block and nothing else.
            - If this is a code-plus-explanation request, give a short explanation and then one fenced Markdown code block.
            - Any code example must use concrete numeric values or clearly defined variables, without ellipses or invalid placeholder identifiers.
            - If more than one concept is asked, cover all of them explicitly.

            {code_repair_rules or ''}

            Return strict JSON only.
            """
        ).strip()

        raw_content = self.kb._llm_json_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0 if task_profile.get("is_code_request") else 0.18,
        )
        return self.kb._parse_llm_answer_payload(
            raw_content,
            query=resolved_query,
            task_profile=task_profile,
            allowed_source_ids=allowed_source_ids,
        )

    def _generate_unified_payload(
        self,
        *,
        raw_query: str,
        resolved_query: str,
        history: Sequence[Dict[str, str]],
        resolution_context: Dict[str, Any],
        retrieval: Dict[str, Any],
    ) -> Dict[str, Any]:
        task_profile = dict(retrieval.get("task_profile") or {})
        citations = list(retrieval.get("citations") or [])
        citation_hits = list(retrieval.get("citation_hits") or [])
        related_kps = list(retrieval.get("related_kps") or [])
        coverage_level = clean_display_text(retrieval.get("coverage_level", "")) or "none"
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
        repeat_instruction = self._repeat_instruction(resolved_query, history)
        coverage_question = self._asks_course_coverage(raw_query) or self._asks_course_coverage(resolved_query)

        system_prompt = textwrap.dedent(
            """
            You are the student-facing chat assistant for Course 4186, a commercial computer vision learning platform.
            Return strict JSON only with exactly two keys:
            - answer
            - used_sources

            Core policy:
            - Answer the student's actual question well before thinking about citation packaging.
            - Use the standalone resolved question as the primary task.
            - Use conversation history only to preserve continuity, not to drift away from the current request.
            - If the topic is within computer vision, answer it even when lecture coverage is partial or absent.
            - Use course sources only when they genuinely support the final answer.
            - Never mention source ids, retrieval, file names, page numbers, or system internals inside the answer body.
            - Keep the tone natural, student-facing, and commercially polished rather than robotic or demo-like.
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

            Course-coverage question:
            {str(bool(coverage_question)).lower()}

            Allowed source ids:
            {", ".join(allowed_source_ids) or 'None'}

            Evidence:
            {evidence_text}

            Variation requirement:
            {repeat_instruction}

            Output rules:
            - Answer the standalone question directly and completely.
            - If Course-coverage question is true, explicitly state whether the topic is directly covered, only touched on indirectly, or not covered in the course.
            - If this is a code-only request, answer with exactly one fenced Markdown code block and nothing else.
            - If this is a code-plus-explanation request, give a short explanation and then one fenced Markdown code block.
            - For code examples, prefer a small runnable example with concrete numeric inputs or clearly defined variables.
            - Do not use placeholder ellipses, invalid identifiers like x1', or pseudo-code fragments that cannot run.
            - If the student asked about multiple concepts, address all of them explicitly.
            - If coverage level is direct, keep the answer grounded in the lecture evidence and select the genuinely supporting used_sources.
            - If coverage level is related or none, do not force course claims. You may answer from reliable computer-vision knowledge and keep used_sources empty or partial as appropriate.
            - Do not output anything outside the JSON object.
            """
        ).strip()

        raw_content = self.kb._llm_json_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.08 if task_profile.get("is_code_request") else 0.22,
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
        payload = self._autofill_used_sources(payload, retrieval=retrieval)

        repeat_context = self.kb._repeat_answer_context(resolved_query, history)
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
            payload = self._repair_unified_payload(
                raw_query=raw_query,
                resolved_query=resolved_query,
                history_text=history_text,
                session_summary=clean_display_text(resolution_context.get("session_summary", "")),
                active_topic=clean_display_text(resolution_context.get("active_topic", "")),
                task_profile=task_profile,
                coverage_level=coverage_level,
                related_kps=related_kps,
                evidence_text=evidence_text,
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
            payload = self._autofill_used_sources(payload, retrieval=retrieval)
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
            raise ValueError("Unified chat pipeline answer did not pass validation: " + " | ".join(issues))
        return payload

    def _finalize_unified_response(
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
        task_profile = dict(retrieval.get("task_profile") or {})
        citations = list(retrieval.get("citations") or [])
        coverage_level = clean_display_text(retrieval.get("coverage_level", "")) or "none"

        payload = self._autofill_used_sources(payload, retrieval=retrieval)
        used_sources = [
            clean_display_text(item).upper()
            for item in payload.get("used_sources", [])
            if clean_display_text(item)
        ]

        if used_sources:
            final_citations = self.kb._select_final_citations(citations, used_sources, max_count=3)
        elif coverage_level == "direct" and citations:
            target_count = len(task_profile.get("target_concepts") or [])
            if target_count <= 0:
                citation_count = 1
            elif target_count == 1:
                citation_count = 1
            else:
                citation_count = min(3, target_count)
            final_citations = list(citations[:citation_count])
        elif (
            citations
            and clean_display_text(resolution_context.get("resolved_query", "")) != resolved_query
            and coverage_level in {"direct", "related"}
        ):
            final_citations = list(citations[:1])
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
            related=retrieval.get("related_kps") or [],
        )
        return {
            "answer": answer_text,
            "citations": final_citations,
            "related_kps": list(retrieval.get("related_kps") or []),
            "mode": mode,
            "session_memory": session_memory,
            "_resolved_query": resolved_query,
            "_resolution_context": dict(resolution_context or {}),
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
            return self._assistant_response_payload(
                answer_text=special_response,
                raw_query=raw_query,
                history=history,
                previous_memory=session_memory,
                resolved_query=resolved_query,
                resolution_context=resolution_context,
                mode="assistant",
            )

        if not self._is_in_scope(raw_query, resolved_query, resolution_context):
            return self._assistant_response_payload(
                answer_text=self._strict_out_of_scope_response(raw_query, history),
                raw_query=raw_query,
                history=history,
                previous_memory=session_memory,
                resolved_query=resolved_query,
                resolution_context=resolution_context,
                mode="assistant",
            )

        retrieval_query = self._effective_retrieval_query(
            raw_query,
            resolved_query,
            resolution_context,
            history,
            session_memory,
        )

        retrieval = self._refine_retrieval(self._retrieve_context(raw_query, retrieval_query, top_k=top_k))
        task_profile = dict(retrieval.get("task_profile") or {})

        if task_profile.get("is_question_generation_request"):
            practice_response = self.kb._practice_question_response(
                raw_query,
                history,
                retrieval["related_kps"],
                retrieval["citation_hits"][:4],
                retrieval["citations"],
                task_profile=task_profile,
                resolved_query=retrieval_query,
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
            return self._finalize_unified_response(
                raw_query=raw_query,
                history=history,
                resolved_query=retrieval_query,
                resolution_context=resolution_context,
                retrieval=patched_retrieval,
                payload=payload,
                previous_memory=session_memory,
                mode=practice_response.get("mode", "practice_question"),
            )

        if self._asks_for_reference(raw_query):
            payload = self._reference_guidance_payload(
                raw_query=raw_query,
                retrieval=retrieval,
            )
            return self._finalize_unified_response(
                raw_query=raw_query,
                history=history,
                resolved_query=retrieval_query,
                resolution_context=resolution_context,
                retrieval=retrieval,
                payload=payload,
                previous_memory=session_memory,
                mode="reference_v3",
            )

        if task_profile.get("is_code_request"):
            template_payload = build_code_template(
                f"{raw_query}\n{retrieval_query}",
                code_only=bool(task_profile.get("is_code_only_request")),
                variant_index=self._exact_query_repeat_count(raw_query, history),
            )
            if template_payload:
                return self._finalize_unified_response(
                    raw_query=raw_query,
                    history=history,
                    resolved_query=retrieval_query,
                    resolution_context=resolution_context,
                    retrieval=retrieval,
                    payload={"answer": template_payload["answer"], "used_sources": []},
                    previous_memory=session_memory,
                    mode="template_code_v3",
                )

        llm_available = self.kb._llm_client is not None and bool(self.kb._llm_model)
        if llm_available:
            for _ in range(2):
                try:
                    payload = self._generate_unified_payload(
                        raw_query=raw_query,
                        resolved_query=retrieval_query,
                        history=history,
                        resolution_context=resolution_context,
                        retrieval=retrieval,
                    )
                    return self._finalize_unified_response(
                        raw_query=raw_query,
                        history=history,
                        resolved_query=retrieval_query,
                        resolution_context=resolution_context,
                        retrieval=retrieval,
                        payload=payload,
                        previous_memory=session_memory,
                        mode="llm_v3",
                    )
                except Exception:
                    continue

        fallback_payload = self._fallback_payload(
            resolved_query=retrieval_query,
            retrieval=retrieval,
        )
        return self._finalize_unified_response(
            raw_query=raw_query,
            history=history,
            resolved_query=retrieval_query,
            resolution_context=resolution_context,
            retrieval=retrieval,
            payload=fallback_payload,
            previous_memory=session_memory,
            mode="fallback_v3",
        )
