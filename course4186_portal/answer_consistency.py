from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence


SOURCE_ID_TOKEN_RE = re.compile(r"\bS\d+\b", re.IGNORECASE)
USED_SOURCES_FIELD_RE = re.compile(
    r"""(?is)
    ["']?used_sources["']?\s*:\s*
    (?P<list>\[\s*"?(?:S\d+)"?(?:\s*,\s*"?(?:S\d+)"?)*\s*\])
    """
)
BRACKETED_SOURCE_LIST_RE = re.compile(
    r'(?is)\[\s*"?(?:S\d+)"?(?:\s*,\s*"?(?:S\d+)"?)*\s*\]'
)
TRAILING_SOURCE_LIST_RE = re.compile(
    r'(?is)(?P<prefix>.*?)(?P<list>\[\s*"?(?:S\d+)"?(?:\s*,\s*"?(?:S\d+)"?)*\s*\])\s*$'
)
COURSE_SOURCE_LINE_RE = re.compile(r"(?is)(?:\n|\r|\s)*Course source:.*$")
CODE_BLOCK_RE = re.compile(r"(```[\s\S]*?```)")

LECTURE_VERB_RE = re.compile(
    r"\blecture\s+[0-9]+(?:[A-Za-z-]*[0-9A-Za-z-]*)?\s+"
    r"(shows|show|explains|explain|covers|cover|introduces|introduce|describes|describe|details|detail|presents|present)\b",
    re.IGNORECASE,
)
SLIDES_FROM_LECTURES_RE = re.compile(
    r"\b(?:the\s+)?slides?\s+from\s+lectures?\s+"
    r"[0-9][0-9A-Za-z,\-\sandor]*",
    re.IGNORECASE,
)
FROM_LECTURES_RE = re.compile(
    r"\bfrom\s+lectures?\s+[0-9][0-9A-Za-z,\-\sandor]*",
    re.IGNORECASE,
)
IN_LECTURES_RE = re.compile(
    r"\bin\s+lectures?\s+[0-9][0-9A-Za-z,\-\sandor]*",
    re.IGNORECASE,
)
LECTURE_LIST_RE = re.compile(
    r"\blectures?\s+[0-9]+(?:-[0-9A-Za-z]+)?"
    r"(?:\s*,\s*[0-9]+(?:-[0-9A-Za-z]+)?)*"
    r"(?:\s*,?\s*(?:and|or)\s+[0-9]+(?:-[0-9A-Za-z]+)?)?",
    re.IGNORECASE,
)
FILE_REFERENCE_RE = re.compile(
    r"\b[A-Za-z0-9][A-Za-z0-9._-]*\.pdf\b(?:\s*\(\s*page\s*\d+\s*\))?",
    re.IGNORECASE,
)
PAGE_REFERENCE_RE = re.compile(r"\bpage\s+\d+\b", re.IGNORECASE)


def _unique_upper(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    rows: List[str] = []
    for value in values:
        normalized = str(value or "").strip().upper()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        rows.append(normalized)
    return rows


def parse_source_id_list(text: str) -> List[str]:
    return _unique_upper(SOURCE_ID_TOKEN_RE.findall(str(text or "")))


def extract_used_source_ids(raw_text: str) -> List[str]:
    candidate = str(raw_text or "").strip()
    if not candidate:
        return []

    field_match = USED_SOURCES_FIELD_RE.search(candidate)
    if field_match:
        ids = parse_source_id_list(field_match.group("list"))
        if ids:
            return ids

    trailing_match = TRAILING_SOURCE_LIST_RE.search(candidate)
    if trailing_match:
        prefix = str(trailing_match.group("prefix") or "").strip()
        if prefix and re.search(r"[A-Za-z0-9\u4e00-\u9fff]", prefix):
            ids = parse_source_id_list(trailing_match.group("list"))
            if ids:
                return ids

    tail = candidate[-120:]
    bracket_matches = list(BRACKETED_SOURCE_LIST_RE.finditer(tail))
    if bracket_matches:
        ids = parse_source_id_list(bracket_matches[-1].group(0))
        if ids:
            return ids
    return []


def strip_source_id_list_suffix(text: str) -> str:
    candidate = str(text or "").strip()
    if not candidate:
        return ""
    trailing_match = TRAILING_SOURCE_LIST_RE.search(candidate)
    if trailing_match:
        prefix = str(trailing_match.group("prefix") or "").rstrip()
        if prefix and re.search(r"[A-Za-z0-9\u4e00-\u9fff]", prefix):
            return prefix
    return candidate


def strip_course_source_line(text: str) -> str:
    return COURSE_SOURCE_LINE_RE.sub("", str(text or "")).strip()


def _normalize_non_code_segment(text: str) -> str:
    candidate = str(text or "")
    if not candidate:
        return ""

    candidate = LECTURE_VERB_RE.sub(lambda m: f"the lecture materials {m.group(1)}", candidate)
    candidate = SLIDES_FROM_LECTURES_RE.sub("the lecture materials", candidate)
    candidate = FROM_LECTURES_RE.sub("from the lecture materials", candidate)
    candidate = IN_LECTURES_RE.sub("in the lectures", candidate)
    candidate = FILE_REFERENCE_RE.sub("the lecture materials", candidate)
    candidate = LECTURE_LIST_RE.sub("the lectures", candidate)
    candidate = PAGE_REFERENCE_RE.sub("", candidate)

    candidate = re.sub(r"\bslides?\s+from\s+the lectures\b", "the lecture materials", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\bshown in the lectures\b", "shown in the lecture materials", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\bas shown in the lectures\b", "as shown in the lecture materials", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\bfrom the lecture materials\s+materials\b", "from the lecture materials", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\bthe lecture materials\s+materials\b", "the lecture materials", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\bthe the\b", "the", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\(\s*\)", "", candidate)
    candidate = re.sub(r"\s+([,.;:])", r"\1", candidate)
    candidate = re.sub(r"([,.;:]){2,}", r"\1", candidate)
    candidate = re.sub(r"[ \t]{2,}", " ", candidate)
    candidate = re.sub(r"\n{3,}", "\n\n", candidate)
    return candidate.strip()


def normalize_answer_body_sources(text: str) -> str:
    candidate = str(text or "")
    if not candidate:
        return ""

    parts = CODE_BLOCK_RE.split(candidate)
    normalized_parts: List[str] = []
    for part in parts:
        if not part:
            continue
        if part.startswith("```"):
            normalized_parts.append(part.strip())
            continue
        normalized_parts.append(_normalize_non_code_segment(part))

    rebuilt_parts: List[str] = []
    for part in normalized_parts:
        if not part:
            continue
        piece = part if part.startswith("```") else part
        if rebuilt_parts:
            previous = rebuilt_parts[-1]
            if piece.startswith("```") and not previous.endswith("\n"):
                rebuilt_parts[-1] = previous.rstrip() + "\n\n"
            elif previous.rstrip().endswith("```") and not piece.startswith("\n"):
                rebuilt_parts[-1] = previous.rstrip() + "\n\n"
        rebuilt_parts.append(piece)
    rebuilt = "".join(rebuilt_parts)
    rebuilt = re.sub(r"\n{3,}", "\n\n", rebuilt).strip()
    return rebuilt


def citation_course_source_line(citations: Sequence[Dict[str, Any]]) -> str:
    refs: List[str] = []
    for item in list(citations or [])[:3]:
        source_name = str(item.get("display_source") or item.get("source") or "").replace("\\", "/").split("/")[-1].strip()
        location = str(item.get("location") or "").strip()
        if not source_name:
            continue
        refs.append(f"{source_name}, {location}" if location else source_name)
    if not refs:
        return ""
    return "Course source: " + "; ".join(refs) + "."


def rebuild_answer_with_citations(text: str, citations: Sequence[Dict[str, Any]]) -> str:
    body = strip_course_source_line(strip_source_id_list_suffix(str(text or ""))).strip()
    body = normalize_answer_body_sources(body)
    return body
