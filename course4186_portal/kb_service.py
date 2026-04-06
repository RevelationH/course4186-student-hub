from __future__ import annotations

import json
import hashlib
import math
import os
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from ftfy import fix_text as ftfy_fix_text
except Exception:
    ftfy_fix_text = None

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except Exception:
    HuggingFaceEmbeddings = None
    FAISS = None

from env_loader import load_project_env
from course4186_portal.answer_consistency import (
    extract_used_source_ids,
    normalize_answer_body_sources,
    rebuild_answer_with_citations,
    strip_course_source_line,
    strip_source_id_list_suffix,
)

load_project_env()


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_CANDIDATES: List[Path] = []
if os.getenv("COURSE4186_ARTIFACT_DIR", "").strip():
    ARTIFACT_CANDIDATES.append(Path(os.getenv("COURSE4186_ARTIFACT_DIR", "").strip()))
ARTIFACT_CANDIDATES.extend(
    [
        ROOT_DIR / "course4186_rag" / "artifacts_full_course",
        ROOT_DIR / "course4186_rag" / "artifacts_week1_week6",
        ROOT_DIR / "course4186_rag" / "artifacts_dense",
        ROOT_DIR / "course4186_rag" / "artifacts",
    ]
)
DEFAULT_EMBED_MODEL = os.getenv("COURSE4186_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*")
OPTION_RE = re.compile(r"^\s*([A-D])\.\s*(.+?)\s*$")
STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "about", "have",
    "will", "then", "than", "your", "their", "they", "them", "what", "when", "where",
    "which", "while", "using", "used", "mainly", "because", "there", "these", "those",
    "through", "between", "being", "also", "does", "such", "give", "briefly", "state",
    "course", "would", "could", "should", "into", "only", "over", "is", "are", "was",
    "were", "to", "of", "in", "on", "at", "by", "as", "be", "a", "an", "it", "or",
    "do", "did", "done", "can", "just", "please", "tell", "show", "me", "explain",
    "describe", "define", "summarize", "summary",
}
QUERY_NOISE_TOKENS = STOPWORDS | {
    "who", "are", "you", "hello", "hi", "hey", "thanks", "thank", "please", "me",
    "my", "our", "ours", "yourself", "introduce", "today", "tell",
}
QUERY_INTENT_TOKENS = {
    "what", "who", "why", "how", "when", "where", "which", "tell", "show",
    "give", "explain", "describe", "define", "introduce", "overview",
    "about", "example", "examples", "sample", "samples", "code",
}
COURSE_CONTEXT_HINTS = {
    "week", "weeks", "lecture", "lectures", "tutorial", "tutorials", "revision",
    "slide", "slides", "quiz", "report", "reports", "practice", "knowledge",
    "point", "points", "topic", "topics", "material", "materials", "notes",
}
CV_DOMAIN_PHRASES = {
    "computer vision",
    "object detection",
    "object recognition",
    "image classification",
    "image segmentation",
    "semantic segmentation",
    "instance segmentation",
    "optical flow",
    "structure from motion",
    "camera calibration",
    "epipolar geometry",
    "feature matching",
    "stereo vision",
    "3d reconstruction",
    "multi-view geometry",
}
CV_DOMAIN_TOKENS = {
    "image", "images", "video", "videos", "pixel", "pixels", "camera", "cameras",
    "convolution", "filter", "filters", "filtering", "kernel", "edge", "edges",
    "corner", "corners", "keypoint", "keypoints", "feature", "features", "descriptor",
    "descriptors", "matching", "correspondence", "segmentation", "object", "objects",
    "detection", "recognition", "tracking", "reconstruction", "geometry", "epipolar",
    "stereo", "disparity", "depth", "motion", "flow", "optical", "homography",
    "pose", "calibration", "triangulation", "projection", "pinhole", "intrinsic",
    "intrinsics", "extrinsic", "extrinsics", "sfm", "sift", "harris", "cnn",
    "yolo", "bbox", "bounding", "box", "boxes",
}
GREETING_RE = re.compile(r"^\s*(hi|hello|hey|good morning|good afternoon|good evening)\s*[!.?]*\s*$", re.IGNORECASE)
IDENTITY_RE = re.compile(r"\b(who are you|what are you|introduce yourself)\b", re.IGNORECASE)
HELP_RE = re.compile(r"\b(what can you do|how can you help|help me use this system)\b", re.IGNORECASE)
THANKS_RE = re.compile(r"\b(thanks|thank you|appreciate it)\b", re.IGNORECASE)
REPORT_RE = re.compile(r"\b(show|open|view|check).*(learning report|study summary)\b", re.IGNORECASE)
QUIZ_RE = re.compile(r"\b(open|show|start|go to).*(quiz|practice)\b", re.IGNORECASE)
DEFINITION_RE = re.compile(r"\b(what is|what are|define|definition|overview|introduce)\b", re.IGNORECASE)
COMPARISON_RE = re.compile(
    r"\b("
    r"compare|comparison|different|difference|distinguish|contrast|versus|vs\.?|"
    r"similarities? between|differences? between|compared with|compared to|as opposed to|rather than"
    r")\b",
    re.IGNORECASE,
)
RELATION_RE = re.compile(
    r"\b("
    r"relation|relationship|relate|related|connection|connected|link between|fit together|work together"
    r")\b",
    re.IGNORECASE,
)
WHY_HOW_RE = re.compile(r"^\s*(why|how)\b", re.IGNORECASE)
APPLICATION_RE = re.compile(
    r"\b("
    r"used for|use for|application|applications|example|examples|when do we use|when should we use|"
    r"use case|use cases"
    r")\b",
    re.IGNORECASE,
)
CODE_REQUEST_RE = re.compile(
    r"\b("
    r"code|python|pytorch|snippet|sample code|example code|implementation|implement|write code|show code|demo code|"
    r"program|script|conv1d|conv2d|torch\.nn|torch\.nn\.functional"
    r")\b",
    re.IGNORECASE,
)
COURSE_QUERY_ALIAS_RULES = (
    (re.compile(r"\bsfm\b", re.IGNORECASE), "structure from motion"),
    (re.compile(r"\bstructure[- ]from[- ]motion\b", re.IGNORECASE), "structure from motion"),
    (re.compile(r"\bcv\b", re.IGNORECASE), "computer vision"),
)
DISPLAY_REPLACEMENTS = {
    "\u2022": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2032": "'",
    "\u00d7": "x",
    "\u2212": "-",
    "\u22a4": "T",
    "\u03bb": "lambda",
    "\u00a0": " ",
    "\uf06c": "lambda",
    "\uf0b7": "-",
    "\ufffd": "",
    "\u0ddc": "",
    "\u0ddd": "",
    "\U0001d465": "x",
}
CORE_COURSE_TOPICS = {
    "computer vision",
    "image filtering",
    "local features",
    "harris corner",
    "harris corners",
    "corner detection",
    "convolution",
    "correlation",
    "edge detection",
    "image resampling",
    "texture analysis",
    "color histogram",
    "bag of words",
    "bag-of-words",
    "image retrieval",
    "geometric transformations",
    "camera model",
    "camera geometry",
    "camera projection",
    "pinhole camera",
    "projection",
    "epipolar geometry",
    "epipolar constraint",
    "essential matrix",
    "fundamental matrix",
    "stereo vision",
    "disparity",
    "rectification",
    "optical flow",
    "aperture problem",
    "lucas-kanade",
    "sift",
    "harris",
    "feature matching",
    "image alignment",
    "homography",
    "cnn",
    "segmentation",
    "structure from motion",
    "sfm",
    "cross product",
    "dot product",
    "camera calibration",
    "calibration",
    "triangulation",
    "bundle adjustment",
    "reprojection error",
    "reprojection",
    "intrinsic parameters",
    "extrinsic parameters",
    "intrinsics",
    "extrinsics",
    "pose estimation",
    "3d reconstruction",
    "multi-view geometry",
}
WEAK_COURSE_QUERY_TOKENS = {
    "new", "database", "normalization", "model", "models", "matching", "match",
    "query", "queries", "classification", "classify", "search", "retrieval",
    "image", "images", "vision", "feature", "features", "motion", "large",
    "scale", "result", "results", "application", "applications", "reading",
    "draft", "introduction", "overview", "basic", "concept", "concepts",
}
STRONG_SINGLE_TERM_ANCHORS = {
    "convolution", "correlation", "epipolar", "essential", "fundamental",
    "sift", "harris", "homography", "segmentation", "resampling", "aliasing",
    "stereo", "disparity", "rectification", "optical", "gaussian", "kernel",
    "lucas", "kanade", "occlusion", "pinhole", "texture", "corners",
    "descriptor", "descriptors", "convolutional", "cnn", "sfm",
    "triangulation", "calibration", "reprojection", "intrinsic",
    "intrinsics", "extrinsic", "extrinsics", "pose", "bundle",
}
DEFINITION_TITLE_HINTS = {
    "goal", "overview", "introduction", "what is", "why study", "motivation",
    "definition", "basic idea", "key idea", "camera model", "epipolar",
}
LOW_SIGNAL_TITLE_HINTS = {
    "results", "applications", "draft", "appendix", "reading", "summary",
}
LOW_QUALITY_VARIANT_PATTERNS = [
    re.compile(r"\bwhich\s+(keyword|knowledge point|topic|lecture|week)\b", re.IGNORECASE),
    re.compile(r"\b(matches?|matching)\s+the\s+following\s+(lecture|summary|study note)\b", re.IGNORECASE),
    re.compile(r"\ball of the above\b", re.IGNORECASE),
    re.compile(r"\bnone of the above\b", re.IGNORECASE),
]
EXPLANATION_META_PATTERNS = [
    re.compile(r"\blecture excerpt\b", re.IGNORECASE),
    re.compile(r"\blecture slide\b", re.IGNORECASE),
    re.compile(r"\blecture material\b", re.IGNORECASE),
    re.compile(r"\boption\s+[A-D]\b", re.IGNORECASE),
    re.compile(r"\bcorrect answer\b", re.IGNORECASE),
    re.compile(r"\bincorrect\b", re.IGNORECASE),
    re.compile(r"\bas (?:mentioned|described|stated|shown)\b", re.IGNORECASE),
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def expand_token_forms(token: str) -> set[str]:
    token = clean_display_text(token).lower()
    if not token:
        return set()
    forms = {token}
    if len(token) > 3:
        forms.add(token + "s")
    if token.endswith("s") and len(token) > 4:
        forms.add(token[:-1])
    if token.endswith("es") and len(token) > 4:
        forms.add(token[:-2])
        forms.add(token[:-1])
    if token.endswith("ed") and len(token) > 4:
        forms.add(token[:-2])
        forms.add(token[:-1])
    if token.endswith("ing") and len(token) > 5:
        forms.add(token[:-3])
        forms.add(token[:-3] + "e")
    if token.endswith("ies") and len(token) > 5:
        forms.add(token[:-3] + "y")
    return {item for item in forms if len(item) > 2}


def safe_snippet(text: str, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", (text or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def repair_text_encoding(text: str) -> str:
    repaired = str(text or "")
    if not repaired:
        return ""
    if ftfy_fix_text is not None:
        try:
            repaired = ftfy_fix_text(repaired)
        except Exception:
            pass
    return repaired


def strip_trailing_copy_markers(text: str) -> str:
    return re.sub(r"(?:\s*\(\d+\))+$", "", str(text or "")).strip()


def clean_display_text(text: str) -> str:
    cleaned = repair_text_encoding(text)
    for source, target in DISPLAY_REPLACEMENTS.items():
        cleaned = cleaned.replace(source, target)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_markdown_text(text: str) -> str:
    cleaned = repair_text_encoding(text)
    for source, target in DISPLAY_REPLACEMENTS.items():
        cleaned = cleaned.replace(source, target)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    parts = re.split(r"(```[\s\S]*?```)", cleaned)
    normalized_parts: List[str] = []
    for part in parts:
        if not part:
            continue
        if part.startswith("```"):
            normalized_parts.append(part.strip())
            continue

        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in part.split("\n")]
        compact_lines: List[str] = []
        blank_run = 0
        for line in lines:
            if not line:
                blank_run += 1
                if blank_run <= 2:
                    compact_lines.append("")
                continue
            blank_run = 0
            compact_lines.append(line)
        normalized_parts.append("\n".join(compact_lines).strip())

    cleaned = "\n\n".join(part for part in normalized_parts if part.strip())
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def looks_like_code_snippet(text: str) -> bool:
    candidate = str(text or "").strip()
    if len(candidate) < 24:
        return False
    code_signals = [
        r"\bimport\b",
        r"\bfrom\b",
        r"\bdef\b",
        r"\bclass\b",
        r"\breturn\b",
        r"\bprint\s*\(",
        r"\btorch\.",
        r"\bnn\.",
        r"\bconv[12]d\b",
        r"=",
        r"\(",
        r"\)",
    ]
    matched = sum(1 for pattern in code_signals if re.search(pattern, candidate))
    return matched >= 3


def guess_code_language(query: str, code: str) -> str:
    query_lower = clean_display_text(query).lower()
    code_lower = str(code or "").lower()
    if "python" in query_lower or any(token in code_lower for token in ("import ", "torch.", "nn.", "def ", "print(")):
        return "python"
    if "javascript" in query_lower or "typescript" in query_lower:
        return "javascript"
    return ""


def promote_inline_code_blocks(text: str, *, query: str = "") -> str:
    body = clean_markdown_text(text)
    if not body or "```" in body:
        return body

    preferred_language = "python" if "python" in clean_display_text(query).lower() or "pytorch" in clean_display_text(query).lower() else ""

    def repl(match: re.Match[str]) -> str:
        code = str(match.group(1) or "").strip()
        if not looks_like_code_snippet(code):
            return match.group(0)
        language = preferred_language or guess_code_language(query, code)
        fence = f"```{language}\n{code}\n```" if language else f"```\n{code}\n```"
        return f"\n\n{fence}\n\n"

    body = re.sub(r"`([^`\n]{24,})`", repl, body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body


def markdown_has_prose_outside_code(text: str) -> bool:
    candidate = str(text or "")
    if not candidate:
        return False
    candidate = re.sub(r"```[\s\S]*?```", " ", candidate)
    candidate = re.sub(r"(?im)^Course source:.*$", " ", candidate)
    candidate = clean_markdown_text(candidate)
    return bool(re.search(r"[A-Za-z]", candidate))


def split_sentences(text: str) -> List[str]:
    candidate = clean_markdown_text(text)
    if not candidate:
        return []
    parts = re.split(r"(?<=[.!?])\s+", candidate)
    return [part.strip() for part in parts if part and part.strip()]


def normalized_path_key(value: str) -> str:
    return clean_display_text(str(value or "")).replace("\\", "/").strip().lower()


def source_family_key(relative_path: str) -> str:
    normalized = normalized_path_key(relative_path)
    if not normalized:
        return ""
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return ""
    stem = parts[-1].rsplit(".", 1)[0]
    stem = strip_trailing_copy_markers(stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return "/".join(parts[:-1] + [stem])


def canonical_display_source_name(value: str) -> str:
    cleaned = clean_display_text(value).replace("\\", "/")
    name = Path(cleaned).name or cleaned
    source_path = Path(name)
    stem = strip_trailing_copy_markers(source_path.stem)
    suffix = source_path.suffix.lower()
    if suffix in {".ppt", ".pptx", ".doc", ".docx"}:
        suffix = ".pdf"
    if stem and suffix:
        return f"{stem}{suffix}"
    return stem or name or cleaned


def display_unit_type(display_source: str, unit_type: str) -> str:
    suffix = Path(clean_display_text(display_source or "")).suffix.lower()
    normalized_unit_type = str(unit_type or "").lower()
    if suffix == ".pdf" and normalized_unit_type == "slide":
        return "page"
    return normalized_unit_type


def format_source_location(source_type: str, unit_type: str, unit_index: Any) -> str:
    try:
        index = int(unit_index)
    except Exception:
        index = 0
    source_kind = (source_type or "").lower()
    unit_kind = (unit_type or "").lower()

    if unit_kind == "page" or source_kind == "pdf":
        return f"page {index}" if index > 0 else "page"
    if unit_kind == "slide" or source_kind in {"pptx", "ppt"}:
        return f"slide {index}" if index > 0 else "slide"
    if unit_kind == "document" or source_kind == "docx":
        return f"document {index}" if index > 1 else "document"
    return f"section {index}" if index > 0 else "section"


def display_source_location(
    source: str,
    source_type: str,
    unit_type: str,
    unit_index: Any,
    display_source: str = "",
) -> str:
    resolved_display_source = canonical_display_source_name(display_source or source)
    resolved_unit_type = display_unit_type(resolved_display_source, unit_type)
    resolved_source_type = "pdf" if resolved_unit_type == "page" else source_type
    return format_source_location(resolved_source_type, resolved_unit_type, unit_index)


def student_facing_source_name(value: str) -> str:
    display_name = canonical_display_source_name(value)
    stem = Path(display_name).stem if Path(display_name).suffix else display_name
    cleaned = clean_display_text(stem)
    cleaned = re.sub(r"(?:\s*\(\d+\))+$", "", cleaned).strip()
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"([A-Za-z])(\d)", r"\1 \2", cleaned)
    cleaned = re.sub(r"(\d)([A-Za-z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "lecture material"


def student_facing_source_reference(citation: Dict[str, Any]) -> str:
    source_name = student_facing_source_name(citation.get("display_source") or citation.get("source") or "")
    location = clean_display_text(citation.get("location") or "")
    if location:
        location = re.sub(r"(?i)^slide\b", "page", location).strip()
    return f"{source_name}, {location}" if location else source_name


def parse_options(options: Sequence[str]) -> List[Dict[str, str]]:
    parsed: List[Dict[str, str]] = []
    for option in options:
        option = clean_display_text(option or "")
        match = OPTION_RE.match(option)
        if match:
            parsed.append({"label": match.group(1), "text": match.group(2)})
        elif option:
            parsed.append({"label": "", "text": str(option)})
    return parsed


COURSE_SOURCE_TOKEN_RE = re.compile(r"@@COURSE_SOURCE_\d+@@")
USED_SOURCES_RE = re.compile(r"(?is)(?:^|\n)\s*[\"']?used_sources[\"']?\s*:\s*\[[^\]]*\].*$")
TRANSPORT_JSON_RE = re.compile(r"(?is)^\s*\{[\s\S]*?[\"']answer[\"']\s*:")
TRANSPORT_JSON_CODEBLOCK_RE = re.compile(r"(?is)\n```json\b[\s\S]*$")
TRANSPORT_JSON_SUFFIX_RE = re.compile(r"(?is)(?:\n|^)\s*\{\s*[\"']?answer[\"']?\s*:[\s\S]*$")


def parse_json_response(raw_text: str) -> Any:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("Empty JSON response.")

    candidates: List[str] = [text]
    if text.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
        candidates.append(stripped.strip())

    decoder = json.JSONDecoder()
    seen: set[str] = set()
    for candidate in candidates:
        candidate = str(candidate or "").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        for start_index in [0] + [match.start() for match in re.finditer(r"[\{\[]", candidate)]:
            snippet = candidate[start_index:].strip()
            if not snippet or snippet in seen:
                continue
            seen.add(snippet)
            try:
                parsed, _ = decoder.raw_decode(snippet)
                return parsed
            except Exception:
                continue
    raise ValueError("No JSON object found in model response.")


def sanitize_transport_answer_text(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if not text:
        return ""

    if re.match(r"^```json\b", text, flags=re.IGNORECASE):
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```$", "", text).strip()

    text = COURSE_SOURCE_TOKEN_RE.sub("", text)
    text = re.sub(r"(?is),?\s*[\"']?used_sources[\"']?\s*:\s*\[[^\]]*\]\s*\}?$", "", text).strip()
    text = USED_SOURCES_RE.sub("", text).strip()
    text = re.sub(r"(?is)^\s*\{\s*[\"']?answer[\"']?\s*:\s*", "", text).strip()
    text = re.sub(r"(?is)\}\s*$", "", text).strip()

    cut_points: List[int] = []
    for pattern in (TRANSPORT_JSON_CODEBLOCK_RE, TRANSPORT_JSON_SUFFIX_RE):
        match = pattern.search(text)
        if match:
            prefix = text[: match.start()].strip()
            if prefix and re.search(r"[A-Za-z0-9\u4e00-\u9fff]", prefix):
                cut_points.append(match.start())
    if cut_points:
        text = text[: min(cut_points)].rstrip()

    text = strip_source_id_list_suffix(text)
    text = text.strip().strip(",").strip().strip("\"'")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return clean_markdown_text(text)


def contains_transport_artifacts(text: str) -> bool:
    candidate = str(text or "")
    if not candidate:
        return False
    return bool(
        COURSE_SOURCE_TOKEN_RE.search(candidate)
        or USED_SOURCES_RE.search(candidate)
        or re.search(r"(?is)^\s*\{\s*[\"']?answer[\"']?\s*:", candidate)
        or TRANSPORT_JSON_CODEBLOCK_RE.search(candidate)
        or (TRANSPORT_JSON_SUFFIX_RE.search(candidate) and not re.match(r"(?is)^\s*\{\s*[\"']?answer[\"']?\s*:", candidate))
        or re.search(r"(?is)[\"']?used_sources[\"']?\s*:", candidate)
        or bool(extract_used_source_ids(candidate))
    )


def extract_student_answer_text(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if not text:
        return ""

    try:
        parsed = parse_json_response(text)
        if isinstance(parsed, dict):
            answer_text = sanitize_transport_answer_text(parsed.get("answer", ""))
            if answer_text:
                return answer_text
    except Exception:
        pass

    if re.match(r"^```json\b", text, flags=re.IGNORECASE):
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```$", "", text).strip()

    json_answer_match = re.search(r'(?is)"answer"\s*:\s*"((?:\\.|[^"])*)"', text)
    if json_answer_match:
        try:
            return sanitize_transport_answer_text(json.loads(f"\"{json_answer_match.group(1)}\""))
        except Exception:
            return sanitize_transport_answer_text(json_answer_match.group(1))

    labeled_answer_match = re.search(
        r"(?is)\banswer\s*:\s*(.+?)(?:\n\s*[\"']?used_sources[\"']?\s*:|\Z)",
        text,
    )
    if labeled_answer_match:
        candidate = labeled_answer_match.group(1).strip().strip(",")
        candidate = candidate.strip().strip("{}").strip().strip("\"'")
        return sanitize_transport_answer_text(candidate)

    cleaned = re.sub(r"(?is)(?:\n|\r|\s)*Course source:.*$", "", text).strip()
    return sanitize_transport_answer_text(cleaned)


def strip_citation_markers(text: str) -> str:
    cleaned = str(text or "")
    if not cleaned:
        return ""
    cleaned = re.sub(r'[ \t]*\[(?:\s*"?S\d+"?\s*,?)+\][ \t]*', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'[ \t]*\[(?:S\d+)\][ \t]*', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'(?:[ \t]*\[(?:S\d+)\])+', ' ', cleaned, flags=re.IGNORECASE)
    return clean_markdown_text(cleaned)


def lexical_rank(
    query: str,
    items: Sequence[Any],
    text_getter,
    top_k: int,
) -> List[Tuple[float, Any]]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    tokenized_docs = [tokenize(text_getter(item)) for item in items]
    if not tokenized_docs:
        return []

    doc_freq: Counter[str] = Counter()
    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))

    doc_count = len(tokenized_docs)
    avg_len = sum(len(tokens) for tokens in tokenized_docs) / max(doc_count, 1)
    avg_len = max(avg_len, 1.0)
    hits: List[Tuple[float, Any]] = []

    for item, tokens in zip(items, tokenized_docs):
        if not tokens:
            continue
        term_freq = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for term in set(query_tokens):
            freq = term_freq.get(term, 0)
            if freq == 0:
                continue
            idf = math.log(1 + (doc_count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = freq + 1.5 * (1 - 0.75 + 0.75 * doc_len / avg_len)
            score += idf * (freq * 2.5 / denom)
        if score > 0:
            hits.append((score, item))

    hits.sort(key=lambda pair: pair[0], reverse=True)
    return hits[:top_k]


class Course4186KnowledgeBase:
    def __init__(self, artifact_dir: Optional[Path] = None, embedding_model: str = DEFAULT_EMBED_MODEL):
        self.embedding_model = embedding_model
        self.artifact_dir = Path(artifact_dir) if artifact_dir else self._pick_artifact_dir()
        self.kps: List[Dict[str, Any]] = load_json(self.artifact_dir / "knowledge_points.json")
        self.questions: List[Dict[str, Any]] = load_json(self.artifact_dir / "questions.json")
        self.chunks: List[Dict[str, Any]] = load_jsonl(self.artifact_dir / "chunks.jsonl")

        self.kps = [self._clean_payload(row) for row in self.kps]
        self.questions = [self._clean_payload(row) for row in self.questions]
        self.chunks = [self._clean_payload(row) for row in self.chunks]

        self.kp_by_id = {item["kp_id"]: item for item in self.kps}
        self.chunk_by_id = {item["chunk_id"]: item for item in self.chunks}
        self.chunk_search_blobs: Dict[str, str] = {}
        self.chunk_term_sets: Dict[str, set[str]] = {}
        self.chunk_title_term_sets: Dict[str, set[str]] = {}
        self.chunks_by_source_unit: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
        for chunk in self.chunks:
            blob = clean_display_text(
                "\n".join([chunk.get("title", ""), chunk.get("relative_path", ""), chunk.get("text", "")])
            ).lower()
            self.chunk_search_blobs[chunk["chunk_id"]] = blob
            self.chunk_term_sets[chunk["chunk_id"]] = set(tokenize(blob))
            self.chunk_title_term_sets[chunk["chunk_id"]] = set(
                tokenize(clean_display_text("\n".join([chunk.get("title", ""), chunk.get("relative_path", "")])).lower())
            )
            unit_type = str(chunk.get("unit_type", "")).lower()
            try:
                unit_index = int(chunk.get("unit_index", 0) or 0)
            except Exception:
                unit_index = 0
            self.chunks_by_source_unit[(normalized_path_key(chunk.get("relative_path", "")), unit_type, unit_index)].append(chunk)
        self.questions_by_kp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for question in self.questions:
            enriched = dict(question)
            enriched["parsed_options"] = parse_options(question.get("options", []))
            enriched["review_refs"] = self._dedupe_review_refs(
                [self._normalize_review_ref(ref) for ref in question.get("review_refs", [])]
            )
            self.questions_by_kp[question["kp_id"]].append(enriched)
        self.course_terms = self._build_course_terms()
        self.course_anchor_phrases = self._build_course_anchor_phrases()
        self.course_anchor_tokens = self._build_course_anchor_tokens()
        self.kp_term_sets: Dict[str, set[str]] = {}
        for kp in self.kps:
            self.kp_term_sets[kp["kp_id"]] = {
                token
                for token in tokenize(
                    "\n".join([kp.get("name", ""), kp.get("description", ""), " ".join(kp.get("keywords", []))])
                )
                if len(token) > 1 and token not in QUERY_NOISE_TOKENS
            }

        self._dense_enabled = False
        self._vector_store = None
        self._embeddings = None
        self._llm_client = None
        self._llm_model = None
        self._setup_dense_stack()
        self._setup_llm()

    def _clean_payload(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._clean_payload(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._clean_payload(item) for item in value]
        if isinstance(value, str):
            return clean_display_text(value)
        return value

    def _normalize_review_ref(self, ref: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(ref or {})
        source = clean_display_text(item.get("source", ""))
        source_type = clean_display_text(item.get("source_type", ""))
        unit_type = clean_display_text(item.get("unit_type", ""))
        try:
            unit_index = int(item.get("unit_index", 0) or 0)
        except Exception:
            unit_index = 0
        display_source = canonical_display_source_name(item.get("display_source") or source)
        resolved_unit_type = display_unit_type(display_source, unit_type)
        item["source"] = source
        item["display_source"] = display_source
        item["unit_type"] = resolved_unit_type
        item["unit_index"] = unit_index
        item["location"] = display_source_location(
            source=source,
            source_type=source_type,
            unit_type=unit_type,
            unit_index=unit_index,
            display_source=display_source,
        )
        return item

    def _pick_artifact_dir(self) -> Path:
        for candidate in ARTIFACT_CANDIDATES:
            if (candidate / "knowledge_points.json").exists() and (candidate / "chunks.jsonl").exists():
                return candidate
        raise FileNotFoundError("No Course 4186 artifacts found. Build the knowledge base first.")

    def _build_course_terms(self) -> set[str]:
        terms: set[str] = set(COURSE_CONTEXT_HINTS)
        for kp in self.kps:
            source_text = "\n".join(
                [kp.get("name", ""), kp.get("description", ""), " ".join(kp.get("keywords", []))]
            )
            for token in tokenize(source_text):
                if len(token) <= 2:
                    continue
                if token in QUERY_NOISE_TOKENS:
                    continue
                terms.add(token)
        return terms

    def _build_course_anchor_phrases(self) -> set[str]:
        phrases = {clean_display_text(topic).lower() for topic in CORE_COURSE_TOPICS if clean_display_text(topic)}
        for kp in self.kps:
            candidates = [kp.get("name", "")] + list(kp.get("keywords", []))
            for candidate in candidates:
                phrase = clean_display_text(candidate).lower()
                if len(phrase) >= 4:
                    phrases.add(phrase)
        return {phrase for phrase in phrases if phrase}

    def _build_course_anchor_tokens(self) -> set[str]:
        tokens: set[str] = set(STRONG_SINGLE_TERM_ANCHORS)
        for phrase in self.course_anchor_phrases:
            for token in tokenize(phrase):
                if len(token) <= 2:
                    continue
                if token in QUERY_NOISE_TOKENS or token in WEAK_COURSE_QUERY_TOKENS:
                    continue
                tokens.add(token)
        return tokens

    def _expand_course_aliases(self, text: str) -> str:
        expanded = clean_display_text(text).lower()
        for pattern, canonical in COURSE_QUERY_ALIAS_RULES:
            def repl(match: re.Match) -> str:
                matched = clean_display_text(match.group(0)).lower().strip()
                if canonical in matched:
                    return matched
                return f"{matched} {canonical}".strip()

            expanded = pattern.sub(repl, expanded)
        return re.sub(r"\s+", " ", expanded).strip()

    def _content_tokens(self, query: str) -> List[str]:
        return [
            token for token in tokenize(self._expand_course_aliases(query))
            if len(token) > 1 and token not in QUERY_NOISE_TOKENS
        ]

    def _query_focus_tokens(self, query: str) -> List[str]:
        return [
            token for token in self._content_tokens(query)
            if token not in QUERY_INTENT_TOKENS
        ]

    def _query_focus_phrases(self, query: str) -> set[str]:
        phrases: set[str] = set()
        for text in (clean_display_text(query).lower(), self._expand_course_aliases(query)):
            if not text:
                continue
            tokens = [
                token
                for token in tokenize(text)
                if len(token) > 1 and token not in QUERY_NOISE_TOKENS and token not in QUERY_INTENT_TOKENS
            ]
            if not tokens:
                continue
            joined = " ".join(tokens)
            if len(joined) >= 4:
                phrases.add(joined)
            max_ngram = min(4, len(tokens))
            for n in range(2, max_ngram + 1):
                for index in range(len(tokens) - n + 1):
                    phrase = " ".join(tokens[index : index + n])
                    if len(phrase) >= 4:
                        phrases.add(phrase)
        return phrases

    def _queries_are_similar(self, left: str, right: str) -> bool:
        left_clean = clean_display_text(left).lower()
        right_clean = clean_display_text(right).lower()
        if not left_clean or not right_clean:
            return False
        if left_clean == right_clean:
            return True

        left_expanded = self._expand_course_aliases(left)
        right_expanded = self._expand_course_aliases(right)
        if left_expanded == right_expanded:
            return True

        left_phrases = self._query_focus_phrases(left)
        right_phrases = self._query_focus_phrases(right)
        shared_phrases = left_phrases & right_phrases
        if shared_phrases and max(len(phrase) for phrase in shared_phrases) >= 6:
            return True

        left_tokens = set(self._query_focus_tokens(left) or self._content_tokens(left))
        right_tokens = set(self._query_focus_tokens(right) or self._content_tokens(right))
        if not left_tokens or not right_tokens:
            return False

        overlap = left_tokens & right_tokens
        if not overlap:
            return False

        overlap_ratio = len(overlap) / max(min(len(left_tokens), len(right_tokens)), 1)
        jaccard = len(overlap) / max(len(left_tokens | right_tokens), 1)
        if overlap_ratio >= 0.85:
            return True
        if len(overlap) >= 2 and jaccard >= 0.6:
            return True
        return False

    def _recent_answers_for_similar_queries(
        self,
        query: str,
        history: Sequence[Dict[str, str]],
        *,
        limit: int = 3,
    ) -> List[str]:
        replies: List[str] = []
        turns = list(history or [])
        for index, turn in enumerate(turns):
            if str(turn.get("role") or "").lower() != "user":
                continue
            previous_query = clean_display_text(turn.get("content", ""))
            if not previous_query or not self._queries_are_similar(query, previous_query):
                continue

            assistant_reply = ""
            for follow_turn in turns[index + 1 :]:
                follow_role = str(follow_turn.get("role") or "").lower()
                if follow_role == "assistant":
                    assistant_reply = clean_markdown_text(
                        extract_student_answer_text(str(follow_turn.get("content") or ""))
                    )
                    break
                if follow_role == "user":
                    break
            if assistant_reply:
                replies.append(assistant_reply)
        return replies[-limit:]

    def _answer_opening(self, text: str) -> str:
        sentences = split_sentences(extract_student_answer_text(text))
        if sentences:
            return sentences[0]
        return safe_snippet(extract_student_answer_text(text), 140)

    def _answer_similarity_signature(self, text: str) -> str:
        candidate = extract_student_answer_text(text)
        candidate = strip_citation_markers(candidate)
        candidate = re.sub(r"(?im)^Course source:.*$", " ", candidate).strip()
        candidate = re.sub(r"```[\s\S]*?```", " [code] ", candidate)
        candidate = clean_markdown_text(candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip().lower()
        return candidate

    def _answer_too_similar_to_recent(self, answer: str, recent_answers: Sequence[str]) -> bool:
        candidate_signature = self._answer_similarity_signature(answer)
        if not candidate_signature:
            return False
        candidate_opening = clean_display_text(self._answer_opening(answer)).lower()
        candidate_tokens = {
            token for token in tokenize(candidate_signature)
            if token not in STOPWORDS and len(token) > 2
        }
        for previous in recent_answers:
            previous_signature = self._answer_similarity_signature(previous)
            if not previous_signature:
                continue
            if candidate_signature == previous_signature:
                return True
            if len(candidate_signature) >= 80 and candidate_signature[:160] == previous_signature[:160]:
                return True

            previous_opening = clean_display_text(self._answer_opening(previous)).lower()
            previous_tokens = {
                token for token in tokenize(previous_signature)
                if token not in STOPWORDS and len(token) > 2
            }
            overlap = candidate_tokens & previous_tokens
            overlap_ratio = len(overlap) / max(min(len(candidate_tokens), len(previous_tokens)), 1)
            if candidate_opening and candidate_opening == previous_opening and overlap_ratio >= 0.72:
                return True
        return False

    def _repeat_answer_context(self, query: str, history: Sequence[Dict[str, str]]) -> Dict[str, Any]:
        recent_answers = self._recent_answers_for_similar_queries(query, history, limit=3)
        repeat_count = len(recent_answers)
        recent_openings = [opening for opening in (self._answer_opening(item) for item in recent_answers) if opening]
        variation_styles = [
            "Start with the clean definition, then add the key property or implication the student should remember.",
            "Start from intuition or geometry first, then give the formal name or definition.",
            "Start from how the concept is computed, used, or interpreted in practice, then summarize the main idea.",
            "Write it like quick revision for a quiz: state the idea, then the one exam-relevant point most worth remembering.",
        ]
        return {
            "repeat_count": repeat_count,
            "recent_answers": recent_answers,
            "recent_openings": recent_openings,
            "style_instruction": variation_styles[repeat_count % len(variation_styles)],
        }

    def _ranking_query(self, query: str) -> str:
        tokens = self._content_tokens(query)
        if tokens:
            return " ".join(tokens)
        return clean_display_text(query)

    def _course_term_matches(self, tokens: Sequence[str]) -> set[str]:
        return {token for token in tokens if token in self.course_terms}

    def _definition_request(self, query: str) -> bool:
        return bool(DEFINITION_RE.search(clean_display_text(query).lower()))

    def _query_anchor_details(self, query: str) -> Dict[str, set[str]]:
        lowered = self._expand_course_aliases(query)
        tokens = self._content_tokens(query)
        return {
            "tokens": {token for token in tokens if token in self.course_anchor_tokens},
            "phrases": {phrase for phrase in self.course_anchor_phrases if phrase in lowered},
        }

    def _kp_reference_terms(self, item: Dict[str, Any]) -> Dict[str, set[str]]:
        phrases: set[str] = set()
        tokens: set[str] = set()
        name = clean_display_text(item.get("name", "")).lower()
        if name:
            phrases.add(name)
            if "structure from motion" in name:
                phrases.add("sfm")
            if "image filtering and convolution" in name:
                phrases.add("convolution")
                phrases.add("image filtering")
            if "motion and optical flow" in name:
                phrases.add("optical flow")
            tokens.update(
                token
                for token in tokenize(self._expand_course_aliases(name))
                if len(token) > 2 and token not in STOPWORDS
            )
        for keyword in item.get("keywords", []):
            cleaned = clean_display_text(keyword).lower()
            if not cleaned:
                continue
            if len(cleaned) >= 4:
                phrases.add(cleaned)
            tokens.update(
                token
                for token in tokenize(self._expand_course_aliases(cleaned))
                if len(token) > 2 and token not in STOPWORDS
            )
        kp_terms = self.kp_term_sets.get(item.get("kp_id", ""), set())
        tokens.update(token for token in kp_terms & STRONG_SINGLE_TERM_ANCHORS if len(token) > 2)
        return {"phrases": {phrase for phrase in phrases if phrase}, "tokens": tokens}

    def _kp_query_position(self, query: str, item: Dict[str, Any]) -> int:
        lowered = self._expand_course_aliases(query)
        candidates = list(self._kp_reference_terms(item)["phrases"])
        positions = [lowered.find(candidate) for candidate in candidates if candidate and lowered.find(candidate) >= 0]
        return min(positions) if positions else 10 ** 6

    def _task_relevant_kp_hits(
        self,
        query: str,
        kp_context: Optional[Sequence[Dict[str, Any]]],
        *,
        task_profile: Optional[Dict[str, Any]] = None,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        hits = list(kp_context or [])
        if not hits:
            return []
        task_profile = task_profile or self._query_task_profile(query, hits)
        explicit_hits = list(task_profile.get("explicit_kp_hits") or [])
        selected: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        if explicit_hits:
            explicit_hits.sort(
                key=lambda hit: (
                    self._kp_query_position(query, hit["item"]),
                    -float(hit.get("score", 0.0)),
                )
            )
            for hit in explicit_hits:
                kp_id = hit["item"]["kp_id"]
                if kp_id in seen_ids:
                    continue
                seen_ids.add(kp_id)
                selected.append(hit)
                if len(selected) >= limit:
                    return selected[:limit]

        top_score = float(hits[0].get("score", 0.0))
        needs_multi = bool(task_profile.get("needs_multi_concept_coverage"))
        for hit in hits:
            kp_id = hit["item"]["kp_id"]
            if kp_id in seen_ids:
                continue
            profile = self._kp_query_match_profile(query, hit["item"])
            score = float(hit.get("score", 0.0))
            if needs_multi:
                if (
                    profile["exact_name"]
                    or profile["exact_keyword_hits"]
                    or profile["strong_token_hits"]
                    or score >= max(5.0, top_score * 0.28)
                ):
                    selected.append(hit)
                    seen_ids.add(kp_id)
            else:
                selected.append(hit)
                seen_ids.add(kp_id)
            if len(selected) >= (2 if needs_multi else 1):
                break

        return selected[:limit]

    def _query_task_profile(
        self,
        query: str,
        kp_context: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        cleaned = clean_display_text(query)
        lowered = self._expand_course_aliases(query)
        focus_tokens = self._query_focus_tokens(query)
        kp_hits = list(kp_context or [])
        explicit_hits: List[Dict[str, Any]] = []
        for hit in kp_hits[:3]:
            profile = self._kp_query_match_profile(query, hit["item"])
            if profile["exact_name"] or profile["exact_keyword_hits"] or profile["strong_token_hits"]:
                explicit_hits.append(hit)

        is_comparison = bool(COMPARISON_RE.search(lowered))
        is_relation = bool(RELATION_RE.search(lowered)) and not is_comparison
        is_why_how = bool(WHY_HOW_RE.match(cleaned))
        is_application = bool(APPLICATION_RE.search(lowered))
        is_definition = self._definition_request(query)
        is_code_request = self._query_requests_code(query)

        top_score = float(kp_hits[0].get("score", 0.0)) if kp_hits else 0.0
        second_score = float(kp_hits[1].get("score", 0.0)) if len(kp_hits) > 1 else 0.0
        balanced_top_two = bool(
            len(kp_hits) > 1
            and second_score >= max(5.0, top_score * 0.28)
            and len(focus_tokens) >= 2
        )

        explicit_concept_count = len({hit["item"]["kp_id"] for hit in explicit_hits})
        needs_multi_concept_coverage = bool(
            is_comparison
            or is_relation
            or explicit_concept_count >= 2
            or balanced_top_two
        )
        is_simple_definition = bool(
            is_definition
            and not is_code_request
            and not is_comparison
            and not is_relation
            and not is_why_how
            and not is_application
            and not needs_multi_concept_coverage
            and len(focus_tokens) <= 6
        )
        answer_style = (
            "comparison"
            if is_comparison
            else "relation"
            if is_relation
            else "definition"
            if is_simple_definition
            else "explanation"
        )
        relevant_kp_hits = self._task_relevant_kp_hits(
            query,
            kp_hits,
            task_profile={
                "explicit_kp_hits": explicit_hits,
                "needs_multi_concept_coverage": needs_multi_concept_coverage,
            },
            limit=3,
        )
        return {
            "is_definition": is_definition,
            "is_simple_definition": is_simple_definition,
            "is_comparison": is_comparison,
            "is_relation": is_relation,
            "is_why_how": is_why_how,
            "is_application": is_application,
            "is_code_request": is_code_request,
            "needs_multi_concept_coverage": needs_multi_concept_coverage,
            "explicit_kp_hits": explicit_hits,
            "relevant_kp_hits": relevant_kp_hits,
            "answer_style": answer_style,
        }

    def _task_prompt_guidance(self, task_profile: Dict[str, Any]) -> str:
        if task_profile.get("is_comparison"):
            return (
                "This is a comparison question. Explicitly explain both concepts and state at least one concrete difference "
                "in purpose, input/output, or role in the vision pipeline. Do not answer only one side."
            )
        if task_profile.get("is_relation"):
            return (
                "This is a relationship question. Explain how the named concepts connect, and make sure each concept is addressed explicitly."
            )
        if task_profile.get("is_why_how"):
            return "This is a how/why question. Explain the mechanism or reason, not just the definition."
        if task_profile.get("is_application"):
            return "This is an application question. Explain what the concept is used for and why."
        if task_profile.get("is_simple_definition"):
            return "This is a single-concept definition question. A concise revision-style answer is appropriate."
        return "Answer the student's question directly and make sure all named concepts are covered."

    def _answer_mentions_concept(self, answer: str, item: Dict[str, Any]) -> bool:
        body = self._expand_course_aliases(answer)
        terms = self._kp_reference_terms(item)
        if any(phrase in body for phrase in terms["phrases"] if len(phrase) >= 3):
            return True
        answer_tokens = set(self._content_tokens(answer))
        return bool(answer_tokens & terms["tokens"])

    def _answer_satisfies_task(
        self,
        answer: str,
        query: str,
        task_profile: Dict[str, Any],
    ) -> bool:
        body = normalize_answer_body_sources(extract_student_answer_text(answer))
        if not body:
            return False

        relevant_hits = list(task_profile.get("relevant_kp_hits") or [])
        if task_profile.get("needs_multi_concept_coverage") and len(relevant_hits) >= 2:
            covered = sum(1 for hit in relevant_hits[:2] if self._answer_mentions_concept(body, hit["item"]))
            if covered < 2:
                return False

        if task_profile.get("is_comparison"):
            lowered = clean_display_text(body).lower()
            contrast_markers = (
                "while ", "whereas", "in contrast", "by contrast", "unlike", "different",
                "difference", "compared with", "compared to", "rather than", "on the other hand", "both",
            )
            if not any(marker in lowered for marker in contrast_markers):
                return False
        return True

    def _trim_optional_course_tail(self, answer: str, task_profile: Dict[str, Any]) -> str:
        if not task_profile.get("needs_multi_concept_coverage"):
            return extract_student_answer_text(answer)
        body = extract_student_answer_text(answer)
        if not body or "```" in body:
            return body
        parts = [part.strip() for part in re.split(r"\n{2,}", body) if part and part.strip()]
        if len(parts) <= 1:
            return body
        course_openers = (
            "in the context of our course",
            "in the context of this course",
            "in this course",
            "in our course",
            "in terms of the course",
            "in terms of our course",
            "in the lectures",
            "in our lectures",
        )
        kept: List[str] = []
        for index, part in enumerate(parts):
            lowered = part.lower()
            if index > 0 and any(lowered.startswith(prefix) for prefix in course_openers):
                continue
            kept.append(part)
        return "\n\n".join(kept) if kept else parts[0]

    def _focus_overlap_with_item(self, item: Dict[str, Any], focus_tokens: Sequence[str]) -> int:
        chunk_id = item.get("chunk_id")
        chunk_terms = self.chunk_term_sets.get(chunk_id, set()) if chunk_id else set()
        title_terms = self.chunk_title_term_sets.get(chunk_id, set()) if chunk_id else set()
        return len(set(focus_tokens) & (chunk_terms | title_terms))

    def _focus_phrase_overlap_with_item(self, item: Dict[str, Any], focus_phrases: Sequence[str]) -> bool:
        chunk_id = item.get("chunk_id")
        blob = self.chunk_search_blobs.get(chunk_id, "") if chunk_id else ""
        return any(phrase in blob for phrase in focus_phrases)

    def _looks_cv_request(self, query: str) -> bool:
        lowered = self._expand_course_aliases(query)
        if any(phrase in lowered for phrase in CV_DOMAIN_PHRASES):
            return True
        focus_tokens = self._query_focus_tokens(query)
        if not focus_tokens:
            return False
        domain_hits = {token for token in focus_tokens if token in CV_DOMAIN_TOKENS or token in self.course_terms}
        strong_hits = domain_hits & (STRONG_SINGLE_TERM_ANCHORS | {"camera", "image", "video", "object", "geometry"})
        if len(domain_hits) >= 2:
            return True
        if strong_hits and len(focus_tokens) <= 3:
            return True
        return False

    def _course_coverage_level(self, query: str, chunk_hits: Sequence[Dict[str, Any]]) -> str:
        focus_tokens = self._query_focus_tokens(query)
        focus_token_set = set(focus_tokens)
        focus_phrases = self._query_focus_phrases(query)
        if not focus_token_set and not focus_phrases:
            return "none"

        related = False
        for hit in chunk_hits[:4]:
            item = hit["item"]
            if self._is_low_priority_source(item):
                continue
            score = float(hit.get("score", 0.0))
            title_text = clean_display_text(item.get("title", "")).lower()
            phrase_hit = self._focus_phrase_overlap_with_item(item, focus_phrases)
            title_phrase_hit = any(phrase in title_text for phrase in focus_phrases)
            overlap = self._focus_overlap_with_item(item, focus_tokens)
            if phrase_hit and (title_phrase_hit or score >= 6.0):
                return "direct"
            if len(focus_token_set) >= 2 and overlap >= 2:
                return "direct"
            if phrase_hit or overlap >= 1:
                related = True
        return "related" if related else "none"

    def _retrieval_supports_course_answer(
        self,
        query: str,
        kp_hits: Sequence[Dict[str, Any]],
        chunk_hits: Sequence[Dict[str, Any]],
    ) -> bool:
        focus_tokens = self._query_focus_tokens(query)
        focus_token_set = set(focus_tokens)
        focus_phrases = self._query_focus_phrases(query)
        if not focus_token_set and not focus_phrases:
            return False

        preferred_hits = self._preferred_chunk_hits(query, chunk_hits, limit=3, kp_context=kp_hits)
        for hit in preferred_hits:
            item = hit["item"]
            if self._is_low_priority_source(item):
                continue
            score = float(hit.get("score", 0.0))
            if self._focus_phrase_overlap_with_item(item, focus_phrases) and score >= 5.0:
                return True
            overlap = self._focus_overlap_with_item(item, focus_tokens)
            if overlap >= 2:
                return True
            if len(focus_token_set) == 1 and overlap == 1 and score >= 8.0:
                return True

        for hit in kp_hits[:2]:
            item = hit["item"]
            kp_blob = clean_display_text(
                "\n".join([item.get("name", ""), item.get("description", ""), " ".join(item.get("keywords", []))])
            ).lower()
            if any(phrase in kp_blob for phrase in focus_phrases):
                return True
            kp_terms = self.kp_term_sets.get(item.get("kp_id", ""), set())
            overlap = len(focus_token_set & kp_terms)
            if overlap >= 2:
                return True
            if len(focus_token_set) == 1 and overlap == 1 and float(hit.get("score", 0.0)) >= 7.0:
                return True
        return False

    def _kp_query_match_profile(
        self,
        query: str,
        item: Dict[str, Any],
        anchor_details: Optional[Dict[str, set[str]]] = None,
    ) -> Dict[str, Any]:
        lowered = self._expand_course_aliases(query)
        anchor_details = anchor_details or self._query_anchor_details(query)
        kp_name = clean_display_text(item.get("name", "")).lower()
        keyword_phrases = [
            clean_display_text(keyword).lower()
            for keyword in item.get("keywords", [])
            if clean_display_text(keyword)
        ]
        def specific_keyword(phrase: str) -> bool:
            tokens = [
                token
                for token in tokenize(self._expand_course_aliases(phrase))
                if len(token) > 2 and token not in STOPWORDS
            ]
            if not tokens:
                return False
            if len(tokens) >= 2:
                return True
            return tokens[0] in (STRONG_SINGLE_TERM_ANCHORS | {"sfm", "sift", "homography", "epipolar", "convolution"})
        kp_terms = self.kp_term_sets.get(item.get("kp_id", ""), set())
        strong_query_tokens = anchor_details["tokens"] & STRONG_SINGLE_TERM_ANCHORS
        exact_name = bool(kp_name and kp_name in lowered)
        exact_keyword_hits = [phrase for phrase in keyword_phrases if phrase and specific_keyword(phrase) and phrase in lowered]
        strong_token_hits = sorted(strong_query_tokens & kp_terms)
        return {
            "exact_name": exact_name,
            "exact_keyword_hits": exact_keyword_hits,
            "strong_token_hits": strong_token_hits,
            "has_strong_query_anchor": bool(strong_query_tokens),
        }

    def _definition_friendly_title(self, item: Dict[str, Any]) -> bool:
        title_text = clean_display_text(item.get("title", "")).lower()
        return any(term in title_text for term in DEFINITION_TITLE_HINTS)

    def _is_low_signal_title(self, item: Dict[str, Any]) -> bool:
        title_text = clean_display_text(item.get("title", "")).lower()
        if any(term in title_text for term in LOW_SIGNAL_TITLE_HINTS):
            return True
        return self._generic_title_penalty(item) <= -1.2

    def _generic_title_penalty(self, item: Dict[str, Any]) -> float:
        title_text = clean_display_text(item.get("title", "")).lower()
        relative_path = clean_display_text(item.get("relative_path", "")).replace("\\", "/")
        file_stem = Path(relative_path).stem.lower() if relative_path else ""
        file_stem = re.sub(r"\s*\(\d+\)$", "", file_stem).strip()
        if not title_text:
            return -1.0
        if title_text == "all":
            return -2.0
        if file_stem == "all":
            return -1.6
        if file_stem and title_text == file_stem:
            return -1.25
        if re.fullmatch(r"lecture\s*\d+([ -]\d+)?(?: \(\d+\))*", title_text):
            return -1.4
        return 0.0

    def _anchor_overlap_with_item(self, item: Dict[str, Any], anchor_details: Dict[str, set[str]]) -> int:
        chunk_id = item.get("chunk_id")
        chunk_terms = self.chunk_term_sets.get(chunk_id, set()) if chunk_id else set()
        title_terms = self.chunk_title_term_sets.get(chunk_id, set()) if chunk_id else set()
        return len(anchor_details["tokens"] & (chunk_terms | title_terms))

    def _phrase_overlap_with_item(self, item: Dict[str, Any], anchor_details: Dict[str, set[str]]) -> bool:
        chunk_id = item.get("chunk_id")
        blob = self.chunk_search_blobs.get(chunk_id, "") if chunk_id else ""
        return any(phrase in blob for phrase in anchor_details["phrases"])

    def _preferred_chunk_hits(
        self,
        query: str,
        chunk_hits: Sequence[Dict[str, Any]],
        limit: int = 4,
        kp_context: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        anchor_details = self._query_anchor_details(query)
        strong_query_tokens = anchor_details["tokens"] & STRONG_SINGLE_TERM_ANCHORS
        definition_request = self._definition_request(query)
        focus_tokens = self._query_focus_tokens(query)
        focus_token_set = set(focus_tokens)
        focus_phrases = self._query_focus_phrases(query)
        support_chunk_ids: set[str] = set()
        support_source_files: set[str] = set()
        for hit in (kp_context or [])[:2]:
            kp_item = hit["item"]
            support_chunk_ids.update(str(chunk_id) for chunk_id in kp_item.get("support_chunk_ids", []) if str(chunk_id).strip())
            support_source_files.update(
                normalized_path_key(source)
                for source in kp_item.get("source_files", [])
                if normalized_path_key(source)
            )
        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for index, hit in enumerate(chunk_hits):
            item = hit["item"]
            score = float(hit.get("score", 0.0))
            chunk_id = str(item.get("chunk_id") or "")
            relative_path_key = normalized_path_key(item.get("relative_path", ""))
            title_text = clean_display_text(item.get("title", "")).lower()
            chunk_terms = self.chunk_term_sets.get(chunk_id, set())
            if not self._is_low_priority_source(item):
                score += 5.0
            if chunk_id and chunk_id in support_chunk_ids:
                score += 6.0
            elif relative_path_key and relative_path_key in support_source_files:
                score += 3.4
            strong_anchor_hits = len(strong_query_tokens & chunk_terms) if strong_query_tokens else 0
            if strong_anchor_hits:
                score += strong_anchor_hits * 6.4
            elif strong_query_tokens and relative_path_key not in support_source_files:
                score -= 5.4
            if definition_request and self._definition_friendly_title(item):
                score += 3.0
            if definition_request and any(term in clean_display_text(item.get("title", "")).lower() for term in LOW_SIGNAL_TITLE_HINTS):
                score -= 2.4
            score += self._generic_title_penalty(item)
            score += self._anchor_overlap_with_item(item, anchor_details) * 1.6
            if self._phrase_overlap_with_item(item, anchor_details):
                score += 2.5
            focus_phrase_hit = self._focus_phrase_overlap_with_item(item, focus_phrases)
            focus_title_phrase_hit = any(phrase in title_text for phrase in focus_phrases)
            focus_overlap = self._focus_overlap_with_item(item, focus_tokens)
            if focus_phrase_hit:
                score += 13.5
            if focus_title_phrase_hit:
                score += 5.0
            if len(focus_token_set) >= 2:
                if focus_overlap >= 2:
                    score += focus_overlap * 5.2
                elif focus_overlap == 1 and not focus_phrase_hit:
                    score -= 7.0
            elif len(focus_token_set) == 1 and focus_overlap == 1:
                score += 2.6
            scored.append((score, index, hit))
        scored.sort(key=lambda row: (-row[0], row[1]))
        ranked_hits = [row[2] for row in scored]
        primary_hits: List[Dict[str, Any]] = []
        secondary_hits: List[Dict[str, Any]] = []
        fallback_hits: List[Dict[str, Any]] = []
        for hit in ranked_hits:
            item = hit["item"]
            if self._is_low_priority_source(item):
                fallback_hits.append(hit)
            elif definition_request and self._is_low_signal_title(item):
                secondary_hits.append(hit)
            elif self._is_low_signal_title(item):
                secondary_hits.append(hit)
            else:
                primary_hits.append(hit)

        def take_hits(pool: Sequence[Dict[str, Any]], selected: List[Dict[str, Any]], hard_limit: int) -> None:
            seen_units: set[Tuple[str, str, int]] = {
                (
                    source_family_key(hit["item"].get("relative_path", "")),
                    str(hit["item"].get("unit_type", "")).lower(),
                    int(hit["item"].get("unit_index", 0) or 0),
                )
                for hit in selected
            }
            family_counts: Counter[str] = Counter(source_family_key(hit["item"].get("relative_path", "")) for hit in selected)
            for hit in pool:
                item = hit["item"]
                family = source_family_key(item.get("relative_path", ""))
                unit_type = str(item.get("unit_type", "")).lower()
                try:
                    unit_index = int(item.get("unit_index", 0) or 0)
                except Exception:
                    unit_index = 0
                unit_key = (family, unit_type, unit_index)
                if unit_key in seen_units:
                    continue
                # Keep only one hit per lecture family so the same lecture does not
                # appear twice via both PPT/PDF or near-duplicate extracts.
                max_per_family = 1
                if family and family_counts[family] >= max_per_family:
                    continue
                selected.append(hit)
                seen_units.add(unit_key)
                if family:
                    family_counts[family] += 1
                if len(selected) >= hard_limit:
                    break

        selected: List[Dict[str, Any]] = []
        take_hits(primary_hits, selected, limit)
        if len(selected) >= 2:
            return selected[: min(limit, 3)]
        if len(selected) < limit:
            take_hits(secondary_hits, selected, limit)
        if len(selected) >= 2:
            return selected[: min(limit, 3)]
        if selected:
            selected_families = {
                source_family_key(hit["item"].get("relative_path", ""))
                for hit in selected
            }
            has_alternative_high_priority_family = any(
                source_family_key(hit["item"].get("relative_path", "")) not in selected_families
                for hit in (primary_hits + secondary_hits)
                if not self._is_low_priority_source(hit["item"])
            )
            if not has_alternative_high_priority_family:
                return selected[: min(limit, len(selected))]
        if len(selected) < min(limit, 2):
            take_hits(fallback_hits, selected, limit)
        return selected[:limit]

    def _is_grounded_answer_ready(
        self,
        query: str,
        kp_hits: Sequence[Dict[str, Any]],
        chunk_hits: Sequence[Dict[str, Any]],
    ) -> bool:
        anchor_details = self._query_anchor_details(query)
        preferred_hits = self._preferred_chunk_hits(query, chunk_hits, limit=3, kp_context=kp_hits)
        if not anchor_details["tokens"] and not anchor_details["phrases"]:
            return self._retrieval_supports_course_answer(query, kp_hits, preferred_hits or chunk_hits)

        for hit in preferred_hits:
            item = hit["item"]
            if self._is_low_priority_source(item):
                continue
            if self._phrase_overlap_with_item(item, anchor_details):
                return True
            if self._anchor_overlap_with_item(item, anchor_details) >= 1:
                return True

        for hit in kp_hits[:2]:
            item = hit["item"]
            kp_terms = self.kp_term_sets.get(item["kp_id"], set())
            if anchor_details["tokens"] & kp_terms:
                return True
            kp_blob = clean_display_text(
                "\n".join([item.get("name", ""), item.get("description", ""), " ".join(item.get("keywords", []))])
            ).lower()
            if any(phrase in kp_blob for phrase in anchor_details["phrases"]):
                return True
        return self._retrieval_supports_course_answer(query, kp_hits, preferred_hits or chunk_hits)

    def _source_quality_bonus(self, item: Dict[str, Any], ranking_query: str, definition_request: bool) -> float:
        relative_path = normalized_path_key(item.get("relative_path", ""))
        title_text = clean_display_text(item.get("title", "")).lower()
        source_type = str(item.get("source_type", "")).lower()
        file_stem = Path(relative_path).stem.lower() if relative_path else ""
        bonus = 0.0

        if "revision" in relative_path:
            bonus -= 3.2
        if file_stem == "all":
            bonus -= 2.8
        if "/tutorial/" in f"/{relative_path}/" or relative_path.startswith("tutorial/"):
            bonus -= 1.3
        if re.search(r"(^|[\/_.-])(question|questions)([\/_.-]|$)", relative_path):
            bonus -= 2.0
        if re.search(r"(^|[\/_.-])(answer|answers)([\/_.-]|$)", relative_path):
            bonus -= 1.9
        if source_type in {"pptx", "ppt"}:
            bonus += 0.35
        elif source_type == "pdf":
            bonus += 0.12
        if "transpose" in title_text and "transpose" not in ranking_query:
            bonus -= 1.5
        bonus += self._generic_title_penalty(item)
        if definition_request:
            if self._definition_friendly_title(item):
                bonus += 2.5
            if any(term in title_text for term in LOW_SIGNAL_TITLE_HINTS):
                bonus -= 1.4
            if any(term in title_text for term in ["difficult", "challenge"]) and "difficult" not in ranking_query:
                bonus -= 2.0
        return bonus

    def _is_low_priority_source(self, item: Dict[str, Any]) -> bool:
        relative_path = normalized_path_key(item.get("relative_path", ""))
        file_stem = Path(relative_path).stem.lower() if relative_path else ""
        if "revision" in relative_path:
            return True
        if file_stem == "all":
            return True
        if re.search(r"(^|[\/_.-])(tutorial|tutorials)([\/_.-]|$)", relative_path):
            return True
        if re.search(r"(^|[\/_.-])(question|questions)([\/_.-]|$)", relative_path):
            return True
        if re.search(r"(^|[\/_.-])(answer|answers)([\/_.-]|$)", relative_path):
            return True
        return False

    def _diversify_chunk_hits(self, rows: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        high_priority_rows = [row for row in rows if not self._is_low_priority_source(row["item"])]
        low_priority_rows = [row for row in rows if self._is_low_priority_source(row["item"])]
        ordered_rows = high_priority_rows + low_priority_rows
        primary: List[Dict[str, Any]] = []
        secondary: List[Dict[str, Any]] = []
        seen_units: set[Tuple[str, str, int]] = set()
        family_counts: Counter[str] = Counter()

        for row in ordered_rows:
            item = row["item"]
            family = source_family_key(item.get("relative_path", ""))
            unit_type = str(item.get("unit_type", "")).lower()
            try:
                unit_index = int(item.get("unit_index", 0) or 0)
            except Exception:
                unit_index = 0
            unit_key = (family, unit_type, unit_index)
            if unit_key in seen_units:
                continue
            seen_units.add(unit_key)
            if family and family_counts[family] == 0:
                primary.append(row)
                family_counts[family] += 1
            else:
                secondary.append(row)

        selected = list(primary[:top_k])
        if len(selected) >= top_k:
            return selected[:top_k]

        for row in secondary:
            family = source_family_key(row["item"].get("relative_path", ""))
            if family and family_counts[family] >= 1:
                continue
            selected.append(row)
            if family:
                family_counts[family] += 1
            if len(selected) >= top_k:
                break
        return selected[:top_k]

    def _build_citations(self, chunk_hits: Sequence[Dict[str, Any]], limit: int = 4) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        seen_units: set[Tuple[str, str, int]] = set()
        for hit in chunk_hits:
            item = hit["item"]
            source = clean_display_text(item.get("relative_path", ""))
            source_type = clean_display_text(item.get("source_type", ""))
            unit_type = clean_display_text(item.get("unit_type", ""))
            try:
                unit_index = int(item.get("unit_index", 0) or 0)
            except Exception:
                unit_index = 0
            display_source = canonical_display_source_name(source)
            resolved_unit_type = display_unit_type(display_source, unit_type)
            unit_key = (normalized_path_key(display_source), resolved_unit_type, unit_index)
            if unit_key in seen_units:
                continue
            seen_units.add(unit_key)
            location = display_source_location(
                source=source,
                source_type=source_type,
                unit_type=unit_type,
                unit_index=unit_index,
                display_source=display_source,
            )
            citations.append(
                {
                    "citation_id": f"S{len(citations) + 1}",
                    "source": source,
                    "display_source": display_source,
                    "location": location,
                    "unit_type": resolved_unit_type,
                    "unit_index": unit_index,
                    "chunk_index": item.get("chunk_index", 0),
                    "section": clean_display_text(item.get("title", "")) or location,
                    "snippet": safe_snippet(item.get("text", ""), 220),
                }
            )
            if len(citations) >= limit:
                break
        return citations

    def _support_candidate_rows(
        self,
        kp_context: Optional[Sequence[Dict[str, Any]]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        for rank, hit in enumerate((kp_context or [])[:2]):
            kp_item = hit["item"]
            base_score = float(hit.get("score", 0.0)) + 18.0 - rank
            for offset, chunk_id in enumerate(kp_item.get("support_chunk_ids", [])[: max(limit, 8)]):
                chunk = self.chunk_by_id.get(str(chunk_id))
                if not chunk:
                    continue
                resolved_chunk_id = str(chunk.get("chunk_id") or "")
                if not resolved_chunk_id or resolved_chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(resolved_chunk_id)
                rows.append({"score": round(base_score - offset * 0.08, 3), "item": chunk})
                if len(rows) >= limit:
                    return rows
        return rows

    def _support_rows_for_kp_hit(
        self,
        query: str,
        kp_hit: Dict[str, Any],
        limit: int,
    ) -> List[Dict[str, Any]]:
        top_kp = kp_hit["item"]
        match_profile = self._kp_query_match_profile(query, top_kp)
        query_lower = clean_display_text(query).lower()
        kp_name = clean_display_text(top_kp.get("name", "")).lower()
        kp_keywords = [clean_display_text(keyword).lower() for keyword in top_kp.get("keywords", [])]
        matched = (
            bool(kp_name and kp_name in query_lower)
            or any(keyword and keyword in query_lower for keyword in kp_keywords)
            or match_profile["exact_name"]
            or bool(match_profile["exact_keyword_hits"])
            or bool(match_profile["strong_token_hits"])
        )
        if not matched and float(kp_hit.get("score", 0.0)) < 11.0:
            return []

        support_rows = self._support_candidate_rows([kp_hit], limit=max(limit, 8))
        seen_chunk_ids: set[str] = {
            str(row["item"].get("chunk_id") or "")
            for row in support_rows
            if str(row["item"].get("chunk_id") or "")
        }
        query_terms = set(self._content_tokens(query))
        kp_terms = self.kp_term_sets.get(top_kp.get("kp_id", ""), set())
        residual_terms = {
            variant
            for term in query_terms
            if term not in kp_terms
            for variant in expand_token_forms(term)
        }
        if residual_terms:
            source_keys = {
                normalized_path_key(source)
                for source in top_kp.get("source_files", [])
                if normalized_path_key(source)
            }
            supplemental_rows: List[Dict[str, Any]] = []
            base_score = float(kp_hit.get("score", 0.0)) + 17.0
            for chunk in self.chunks:
                chunk_id = str(chunk.get("chunk_id") or "")
                if not chunk_id or chunk_id in seen_chunk_ids:
                    continue
                if normalized_path_key(chunk.get("relative_path", "")) not in source_keys:
                    continue
                chunk_terms = self.chunk_term_sets.get(chunk_id, set())
                title_terms = self.chunk_title_term_sets.get(chunk_id, set())
                residual_overlap = residual_terms & (chunk_terms | title_terms)
                if not residual_overlap:
                    continue
                supplemental_rows.append(
                    {
                        "score": round(base_score + len(residual_overlap) * 1.1, 3),
                        "item": chunk,
                    }
                )
            if supplemental_rows:
                supplemental_rows = self._rerank_chunk_hits(query, supplemental_rows, [kp_hit], top_k=max(limit, 8))
                for row in supplemental_rows:
                    chunk_id = str(row["item"].get("chunk_id") or "")
                    if not chunk_id or chunk_id in seen_chunk_ids:
                        continue
                    support_rows.append(row)
                    seen_chunk_ids.add(chunk_id)
        if len(support_rows) < 2:
            return []

        anchor_details = self._query_anchor_details(query)
        focused = [
            row for row in support_rows
            if self._phrase_overlap_with_item(row["item"], anchor_details)
            or self._anchor_overlap_with_item(row["item"], anchor_details) >= 1
        ]
        focused.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        if len(focused) >= 2:
            return focused[:limit]
        support_rows.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        if match_profile["exact_name"] or match_profile["exact_keyword_hits"] or match_profile["strong_token_hits"]:
            return support_rows[:limit]
        return support_rows[:limit] if matched else []

    def _focused_kp_support_hits(
        self,
        query: str,
        kp_context: Optional[Sequence[Dict[str, Any]]],
        limit: int,
        task_profile: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not kp_context:
            return []
        task_profile = task_profile or self._query_task_profile(query, kp_context)
        selected_kp_hits = self._task_relevant_kp_hits(query, kp_context, task_profile=task_profile, limit=3)
        if not selected_kp_hits:
            selected_kp_hits = [kp_context[0]]

        max_kps = 2 if task_profile.get("needs_multi_concept_coverage") else 1
        combined_rows: List[Dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        for kp_hit in selected_kp_hits[:max_kps]:
            rows = self._support_rows_for_kp_hit(query, kp_hit, limit=max(limit, 6))
            for row in rows:
                chunk_id = str(row["item"].get("chunk_id") or "")
                if not chunk_id or chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                combined_rows.append(row)
        if not combined_rows:
            return []
        return self._preferred_chunk_hits(query, combined_rows, limit=limit, kp_context=selected_kp_hits)

    def _multi_concept_chunk_hits(
        self,
        query: str,
        chunk_hits: Sequence[Dict[str, Any]],
        *,
        limit: int,
        ) -> List[Dict[str, Any]]:
        if not chunk_hits:
            return []
        anchor_details = self._query_anchor_details(query)
        focus_tokens = [
            token
            for token in self._query_focus_tokens(query)
            if token in anchor_details["tokens"] and token in STRONG_SINGLE_TERM_ANCHORS
        ]
        prioritized_tokens: List[str] = []
        seen_tokens: set[str] = set()
        for token in focus_tokens:
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            prioritized_tokens.append(token)

        selected: List[Dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        for token in prioritized_tokens[:4]:
            for hit in chunk_hits:
                item = hit["item"]
                chunk_id = str(item.get("chunk_id") or "")
                if not chunk_id or chunk_id in seen_chunk_ids:
                    continue
                chunk_terms = self.chunk_term_sets.get(chunk_id, set())
                title_terms = self.chunk_title_term_sets.get(chunk_id, set())
                if token not in chunk_terms and token not in title_terms:
                    continue
                selected.append(hit)
                seen_chunk_ids.add(chunk_id)
                break

        for hit in chunk_hits:
            item = hit["item"]
            chunk_id = str(item.get("chunk_id") or "")
            if not chunk_id or chunk_id in seen_chunk_ids:
                continue
            selected.append(hit)
            seen_chunk_ids.add(chunk_id)
            if len(selected) >= max(limit, 4):
                break
        return self._preferred_chunk_hits(query, selected, limit=limit, kp_context=None)

    def _rerank_kp_hits(self, query: str, hits: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        query_terms = set(self._content_tokens(query))
        ranking_query = self._ranking_query(query).lower()
        anchor_details = self._query_anchor_details(query)
        strong_query_tokens = anchor_details["tokens"] & STRONG_SINGLE_TERM_ANCHORS
        reranked: List[Dict[str, Any]] = []
        for hit in hits:
            item = hit["item"]
            kp_terms = self.kp_term_sets.get(item["kp_id"], set())
            overlap = len(query_terms & kp_terms)
            phrase_bonus = 3.0 if ranking_query and ranking_query in " ".join(sorted(kp_terms)) else 0.0
            match_profile = self._kp_query_match_profile(query, item, anchor_details=anchor_details)
            score = float(hit["score"]) + overlap * 4.0 + phrase_bonus
            score += 10.0 if match_profile["exact_name"] else 0.0
            score += len(match_profile["exact_keyword_hits"]) * 7.5
            score += len(match_profile["strong_token_hits"]) * 12.0
            if strong_query_tokens and not match_profile["strong_token_hits"]:
                score -= 6.5
            if overlap or phrase_bonus or self._course_term_matches(query_terms) or match_profile["exact_name"] or match_profile["exact_keyword_hits"] or match_profile["strong_token_hits"]:
                reranked.append({"score": round(score, 3), "item": item})
        reranked.sort(key=lambda row: row["score"], reverse=True)
        return reranked[:top_k]

    def _rerank_chunk_hits(
        self,
        query: str,
        hits: Sequence[Dict[str, Any]],
        kp_context: Optional[Sequence[Dict[str, Any]]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        query_terms = set(self._content_tokens(query))
        ranking_query = self._ranking_query(query).lower()
        anchor_details = self._query_anchor_details(query)
        strong_query_tokens = anchor_details["tokens"] & STRONG_SINGLE_TERM_ANCHORS
        definition_request = self._definition_request(query)
        kp_terms: set[str] = set()
        kp_weeks: set[str] = set()
        for hit in (kp_context or [])[:2]:
            kp_terms.update(self.kp_term_sets.get(hit["item"]["kp_id"], set()))
            kp_weeks.update(str(week) for week in hit["item"].get("weeks", []) if str(week).strip())

        reranked: List[Dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        for hit in hits:
            item = hit["item"]
            chunk_id = item.get("chunk_id")
            if not chunk_id or chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            blob = self.chunk_search_blobs.get(chunk_id, "")
            chunk_terms = self.chunk_term_sets.get(chunk_id, set())
            title_terms = self.chunk_title_term_sets.get(chunk_id, set())
            overlap = len(query_terms & chunk_terms)
            title_overlap = len(query_terms & title_terms)
            kp_overlap = len(kp_terms & chunk_terms) if kp_terms else 0
            strong_anchor_hits = len(strong_query_tokens & chunk_terms) if strong_query_tokens else 0
            phrase_bonus = 4.0 if ranking_query and ranking_query in blob else 0.0
            source_bonus = 0.0
            if kp_weeks:
                source_bonus += 1.2 if str(item.get("week") or "") in kp_weeks else -0.8
            if strong_anchor_hits:
                source_bonus += strong_anchor_hits * 7.4
            elif strong_query_tokens and phrase_bonus == 0.0 and title_overlap == 0:
                source_bonus -= 6.2
            source_bonus += self._source_quality_bonus(item, ranking_query, definition_request)
            score = (
                float(hit["score"])
                + overlap * 4.5
                + title_overlap * 3.0
                + kp_overlap * 0.8
                + phrase_bonus
                + source_bonus
            )
            if overlap == 0 and title_overlap == 0 and phrase_bonus == 0.0 and kp_overlap == 0:
                continue
            reranked.append({"score": round(score, 3), "item": item})

        reranked.sort(key=lambda row: row["score"], reverse=True)
        return self._diversify_chunk_hits(reranked, top_k=top_k)

    def _follow_up_support_hits(
        self,
        kp: Dict[str, Any],
        *,
        weak_attempts: Optional[Sequence[Dict[str, Any]]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()

        for offset, chunk_id in enumerate(kp.get("support_chunk_ids", [])[:8]):
            chunk = self.chunk_by_id.get(str(chunk_id))
            if not chunk:
                continue
            resolved_chunk_id = str(chunk.get("chunk_id") or "")
            if not resolved_chunk_id or resolved_chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(resolved_chunk_id)
            rows.append({"score": round(32.0 - offset * 0.2, 3), "item": chunk})
            if len(rows) >= limit:
                return rows

        query_parts = [
            clean_display_text(kp.get("name", "")),
            clean_display_text(kp.get("description", "")),
            " ".join(clean_display_text(keyword) for keyword in kp.get("keywords", [])[:6]),
        ]
        for attempt in list(weak_attempts or [])[:2]:
            query_parts.append(clean_display_text(attempt.get("question", "")))
            query_parts.append(clean_display_text(attempt.get("explanation", "")))
        support_query = " ".join(part for part in query_parts if part).strip()
        if not support_query:
            return rows[:limit]

        kp_context = [{"score": 24.0, "item": kp}]
        for hit in self.search_chunks(support_query, top_k=max(limit, 4), kp_context=kp_context):
            chunk = hit["item"]
            resolved_chunk_id = str(chunk.get("chunk_id") or "")
            if not resolved_chunk_id or resolved_chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(resolved_chunk_id)
            rows.append(hit)
            if len(rows) >= limit:
                break
        return rows[:limit]

    def _fallback_variant_explanation(self, kp_name: str, correct_text: str) -> str:
        topic = clean_display_text(kp_name or "this topic")
        answer = clean_display_text(correct_text).rstrip(".")
        if not answer:
            return f"This follows the main lecture idea from {topic}."
        answer_body = answer[0].lower() + answer[1:] if answer else answer
        if re.match(r"^(by|because|using|through|from)\b", answer_body, re.IGNORECASE):
            return f"The lecture's key point in {topic} is that the method works {answer_body}."
        if re.match(r"^(it|they|this|these)\b", answer_body, re.IGNORECASE):
            return f"The lecture's key point in {topic} is that {answer_body}."
        return f"The lecture's key point in {topic} is {answer_body}."

    def _normalize_generated_explanation(
        self,
        explanation: str,
        *,
        kp_name: str,
        correct_text: str,
    ) -> str:
        cleaned = clean_display_text(explanation)
        if not cleaned:
            return self._fallback_variant_explanation(kp_name, correct_text)

        cleaned = re.sub(r'(?is)\bas (?:mentioned|described|stated|shown)\b[^.?!:]*[:]\s*["\'].*?["\']', "", cleaned)
        cleaned = re.sub(r"(?i)\btherefore,?\s*option\s+[A-D]\s+is\s+the\s+correct\s+answer\.?", "", cleaned)
        cleaned = re.sub(r"(?i)\boption\s+[A-D]\s+is\s+the\s+correct\s+answer\.?", "", cleaned)
        cleaned = re.sub(r"(?i)\bthe\s+correct\s+answer\s+is\s+option\s+[A-D]\.?", "", cleaned)
        cleaned = re.sub(r"(?i)\boption\s+[A-D]\s+is\s+incorrect\b[^.?!]*[.?!]?", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
            if sentence.strip()
        ]
        kept: List[str] = []
        for sentence in sentences:
            lowered = sentence.lower()
            if any(pattern.search(lowered) for pattern in EXPLANATION_META_PATTERNS):
                continue
            kept.append(sentence)
            if len(kept) >= 2:
                break

        normalized = " ".join(kept).strip()
        if not normalized and sentences:
            normalized = sentences[0]
            normalized = re.sub(r"(?i)\btherefore\b[:,]?\s*", "", normalized).strip()
        if not normalized:
            normalized = self._fallback_variant_explanation(kp_name, correct_text)

        normalized = normalized.strip().strip('"').strip("'").strip()
        if len(normalized) > 170:
            normalized = self._fallback_variant_explanation(kp_name, correct_text)
        normalized = textwrap.shorten(normalized, width=170, placeholder="...")
        if normalized and normalized[-1] not in ".!?":
            normalized += "."
        return normalized

    def _normalize_follow_up_variant(
        self,
        *,
        kp: Dict[str, Any],
        item: Any,
        review_refs: Sequence[Dict[str, Any]],
        index: int,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None

        question = clean_display_text(item.get("question", ""))
        raw_explanation = clean_display_text(item.get("explanation", ""))
        if not question or not raw_explanation:
            return None
        if any(pattern.search(question) for pattern in LOW_QUALITY_VARIANT_PATTERNS):
            return None

        raw_options = item.get("options") or []
        if not isinstance(raw_options, list) or len(raw_options) != 4:
            return None

        normalized_options: List[str] = []
        for expected_label, raw_option in zip("ABCD", raw_options):
            option_text = clean_display_text(str(raw_option or ""))
            if not option_text:
                return None
            if any(pattern.search(option_text) for pattern in LOW_QUALITY_VARIANT_PATTERNS):
                return None
            match = OPTION_RE.match(option_text)
            option_body = clean_display_text(match.group(2) if match else option_text)
            if not option_body:
                return None
            normalized_options.append(f"{expected_label}. {option_body}")

        parsed_options = parse_options(normalized_options)
        if len(parsed_options) != 4 or any(not option.get("text") for option in parsed_options):
            return None

        correct_option = clean_display_text(item.get("correct_option", "")).upper()
        if correct_option not in {"A", "B", "C", "D"}:
            return None

        option_lookup = {option.get("label", ""): option.get("text", "") for option in parsed_options}
        correct_text = option_lookup.get(correct_option, "")
        if not correct_text:
            return None
        explanation = self._normalize_generated_explanation(
            raw_explanation,
            kp_name=clean_display_text(kp.get("name", "")),
            correct_text=correct_text,
        )

        return {
            "question_id": self._variant_question_id(kp.get("kp_id", ""), question, normalized_options, index=index),
            "kp_id": kp.get("kp_id"),
            "kp_name": kp.get("name"),
            "question_type": "multiple_choice",
            "question": question,
            "options": normalized_options,
            "parsed_options": parsed_options,
            "correct_option": correct_option,
            "reference_answer": f"{correct_option}. {correct_text}",
            "explanation": explanation,
            "review_refs": [dict(ref) for ref in review_refs],
            "image_path": None,
            "image_caption": None,
            "generated": True,
        }

    def _variant_question_id(
        self,
        kp_id: str,
        question: str,
        options: Sequence[str],
        *,
        index: int,
    ) -> str:
        seed = "|".join(
            [
                clean_display_text(kp_id),
                clean_display_text(question),
                *[clean_display_text(option) for option in options],
            ]
        )
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
        return f"{clean_display_text(kp_id) or 'kp'}-variant-{index}-{digest}"

    def _dedupe_review_refs(self, review_refs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        index_by_key: Dict[tuple[str, str], int] = {}
        for ref in review_refs:
            item = dict(ref)
            source = clean_display_text(item.get("source", ""))
            display_source = canonical_display_source_name(item.get("display_source") or source)
            location = clean_display_text(item.get("location", ""))
            key = (display_source.lower(), location.lower())
            existing_index = index_by_key.get(key)
            if existing_index is None:
                index_by_key[key] = len(rows)
                rows.append(item)
                continue
            existing = rows[existing_index]
            existing_source = clean_display_text(existing.get("source", ""))
            if source.lower().endswith(".pdf") and not existing_source.lower().endswith(".pdf"):
                rows[existing_index] = item
        return rows

    def _assistant_intent(self, query: str) -> Optional[str]:
        cleaned = clean_display_text(query)
        lowered = cleaned.lower()

        if GREETING_RE.match(cleaned):
            return "greeting"
        if IDENTITY_RE.search(lowered):
            return "identity"
        if HELP_RE.search(lowered):
            return "help"
        if REPORT_RE.search(lowered):
            return "report"
        if QUIZ_RE.search(lowered):
            return "quiz"
        if THANKS_RE.search(lowered):
            return "thanks"
        return None

    def _assistant_response_for_query(self, query: str, history: Optional[Sequence[Dict[str, str]]] = None) -> Optional[str]:
        intent = self._assistant_intent(query)
        if not intent:
            return None

        if self._llm_client is not None and self._llm_model:
            try:
                return self._llm_assistant_response(query, intent, history or [])
            except Exception:
                pass

        if intent == "greeting":
            return (
                "Hello. You can ask me about Course 4186 lecture ideas, formulas, methods, or quiz practice."
            )
        if intent == "identity":
            return (
                "I’m the learning assistant for Course 4186. I can help with lecture explanations, revision, and quiz preparation."
            )
        if intent == "help":
            return (
                "I can help you review lecture content, explain knowledge points, and guide you to quiz practice in Course 4186."
            )
        if intent == "report":
            return (
                "You can open Learning Report from the left sidebar to review strengths, weak areas, and suggested next questions."
            )
        if intent == "quiz":
            return (
                "You can open Quiz from the left sidebar for topic-based multiple-choice practice in Course 4186."
            )
        if intent == "thanks":
            return "You’re welcome. If you want, ask another Course 4186 question and we can continue."
        return None

    def _looks_course_request(self, query: str) -> bool:
        lowered = self._expand_course_aliases(query)
        tokens = self._content_tokens(query)
        if not tokens:
            return False
        if re.search(r"\b(week|lecture|tutorial|revision|slide|quiz|report|practice)\s*\d*\b", lowered):
            return True
        if self._looks_cv_request(query):
            return True
        anchor_details = self._query_anchor_details(query)
        if any(topic in lowered for topic in CORE_COURSE_TOPICS):
            return True
        if anchor_details["phrases"]:
            return True
        if anchor_details["tokens"] & STRONG_SINGLE_TERM_ANCHORS:
            return True
        if len(anchor_details["tokens"]) >= 2:
            return True
        kp_hits = self.search_knowledge_points(query, top_k=2)
        if kp_hits:
            top_kp = kp_hits[0]["item"]
            kp_overlap = anchor_details["tokens"] & self.kp_term_sets.get(top_kp["kp_id"], set())
            if float(kp_hits[0]["score"]) >= 7.0 and (kp_overlap or anchor_details["phrases"]):
                return True
        chunk_hits = self.search_chunks(query, top_k=4, kp_context=kp_hits)
        if self._retrieval_supports_course_answer(query, kp_hits, chunk_hits):
            return True
        if not anchor_details["tokens"] and not anchor_details["phrases"]:
            return False
        return self._is_grounded_answer_ready(query, kp_hits, chunk_hits)

    def _contextual_query(self, query: str, history: Sequence[Dict[str, str]]) -> str:
        current = clean_display_text(query)
        if not current or not history:
            return current
        lowered = self._expand_course_aliases(current)
        anchor_details = self._query_anchor_details(current)
        has_explicit_anchor = bool(
            anchor_details["phrases"]
            or (anchor_details["tokens"] & STRONG_SINGLE_TERM_ANCHORS)
            or any(topic in lowered for topic in CORE_COURSE_TOPICS)
        )
        if has_explicit_anchor:
            return current
        if len(self._content_tokens(current)) > 18:
            return current

        for turn in reversed(list(history)[-6:]):
            if str(turn.get("role") or "").lower() != "user":
                continue
            previous = clean_display_text(turn.get("content", ""))
            if not previous or previous == current:
                continue
            previous_lower = self._expand_course_aliases(previous)
            previous_anchor = self._query_anchor_details(previous)
            if previous_anchor["phrases"] or (previous_anchor["tokens"] & STRONG_SINGLE_TERM_ANCHORS) or any(topic in previous_lower for topic in CORE_COURSE_TOPICS):
                return f"{previous}\nFollow-up question: {current}"
        return current

    def _out_of_scope_response(self, query: str, history: Optional[Sequence[Dict[str, str]]] = None) -> str:
        prior_out_of_scope = 0
        for turn in history or []:
            if str(turn.get("role") or "").lower() != "user":
                continue
            if not self._looks_course_request(str(turn.get("content") or "")):
                prior_out_of_scope += 1
        variant_index = prior_out_of_scope + (sum(ord(ch) for ch in clean_display_text(query)) % 3)

        if self._llm_client is not None and self._llm_model:
            try:
                generated = self._llm_assistant_response(query, "out_of_scope", history or [])
                return self._normalize_out_of_scope_answer(generated, variant_index=variant_index)
            except Exception:
                pass
        return self._normalize_out_of_scope_answer("", variant_index=variant_index)

    def _normalize_out_of_scope_answer(self, answer: str, *, variant_index: int = 0) -> str:
        refusal_options = [
            "That question is outside the scope of Course 4186, so I won’t answer it directly.",
            "I can’t help with that directly because it is outside Course 4186, so I won’t answer it directly.",
            "That request is not part of Course 4186, so I should not answer it directly.",
        ]
        redirect_options = [
            "If you want, ask about lecture topics such as convolution, SIFT, camera geometry, stereo vision, or optical flow.",
            "If you want to stay within the course, I can help with topics like convolution, feature matching, epipolar geometry, or optical flow.",
            "You can ask instead about course ideas such as image filtering, SIFT, stereo vision, camera geometry, or homography.",
        ]
        refusal = refusal_options[variant_index % len(refusal_options)]
        redirect = redirect_options[variant_index % len(redirect_options)]

        sentences = split_sentences(answer)
        cleaned_sentences: List[str] = []
        for sentence in sentences:
            lowered = sentence.lower()
            if any(
                marker in lowered
                for marker in [
                    "political science",
                    "philosophical",
                    "sociological",
                    "weather",
                    "joke",
                    "those areas",
                    "those topics",
                    "within those areas",
                ]
            ):
                continue
            cleaned_sentences.append(sentence)

        direct_refusal_present = any(
            marker in sentence.lower()
            for sentence in cleaned_sentences
            for marker in [
                "won't answer this directly",
                "won’t answer this directly",
                "won't answer it directly",
                "won’t answer it directly",
                "not answer it directly",
                "not answer this directly",
                "can’t help with that directly",
                "can't help with that directly",
            ]
        )
        redirect_present = any(
            marker in sentence.lower()
            for sentence in cleaned_sentences
            for marker in [
                "convolution",
                "sift",
                "camera geometry",
                "stereo vision",
                "optical flow",
                "epipolar geometry",
                "homography",
                "image filtering",
                "course 4186",
            ]
        )

        output_parts: List[str] = []
        if direct_refusal_present and cleaned_sentences:
            output_parts.append(cleaned_sentences[0])
        else:
            output_parts.append(refusal)
        if redirect_present:
            redirect_sentence = next(
                (
                    sentence
                    for sentence in cleaned_sentences[1:] if any(
                        marker in sentence.lower()
                        for marker in [
                            "convolution",
                            "sift",
                            "camera geometry",
                            "stereo vision",
                            "optical flow",
                            "epipolar geometry",
                            "homography",
                            "image filtering",
                            "course 4186",
                        ]
                    )
                ),
                "",
            )
            output_parts.append(redirect_sentence or redirect)
        else:
            output_parts.append(redirect)
        return " ".join(part.strip() for part in output_parts if part and part.strip())

    def _insufficient_evidence_response(self, query: str, related: Sequence[Dict[str, Any]]) -> str:
        if related:
            topic_list = ", ".join(item["name"] for item in related[:2])
            return (
                "This looks related to Course 4186, but I could not find a direct enough explanation in the current lecture materials. "
                f"Try asking with the exact method, formula, or lecture topic name, for example: {topic_list}."
            )
        return (
            "I could not find a direct enough explanation for that in the current Course 4186 lecture materials. "
            "Try asking with the exact method, formula, or lecture topic name."
        )

    def _looks_brief_concept_request(
        self,
        query: str,
        *,
        task_profile: Optional[Dict[str, Any]] = None,
        kp_hits: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> bool:
        profile = task_profile or self._query_task_profile(query, kp_hits)
        if not profile.get("is_simple_definition"):
            return False
        hits = list(kp_hits or [])
        if not hits:
            return len(self._query_focus_tokens(query)) <= 5
        top_score = float(hits[0].get("score", 0.0))
        second_score = float(hits[1].get("score", 0.0)) if len(hits) > 1 else 0.0
        return top_score >= max(7.0, second_score + 4.5)

    def _query_requests_code(self, query: str) -> bool:
        return bool(CODE_REQUEST_RE.search(clean_display_text(query)))

    def _code_answer_intro(self, query: str, related: Sequence[Dict[str, Any]], chunk_hits: Sequence[Dict[str, Any]]) -> str:
        lowered = clean_display_text(query).lower()
        if "convolution" in lowered:
            return (
                "Below is a simple PyTorch example that turns the lecture idea of convolution "
                "into code by applying a kernel-based layer to an input tensor."
            )
        if related:
            return f"Below is a simple example that translates the lecture idea of {related[0]['name']} into code."
        top_item = chunk_hits[0]["item"] if chunk_hits else {}
        section = clean_display_text(top_item.get("title", ""))
        if section:
            return f"Below is a simple example connected to the lecture section on {section}."
        return "Below is a simple example that translates the lecture idea into code."

    def _code_answer_outro(self, query: str) -> str:
        lowered = clean_display_text(query).lower()
        if "convolution" in lowered:
            return (
                "You can adjust the input shape, channel counts, kernel size, stride, and padding "
                "to match the image setting you want to test."
            )
        return "You can adapt the tensor shape and parameters to match the specific case you want to test."

    def _meaningful_prose_word_count(self, text: str) -> int:
        cleaned = re.sub(r"(?im)^Course source:.*$", " ", str(text or ""))
        cleaned = clean_markdown_text(cleaned)
        return len(re.findall(r"[A-Za-z]+", cleaned))

    def _ensure_code_answer_explanation(
        self,
        answer: str,
        query: str,
        related: Sequence[Dict[str, Any]],
        chunk_hits: Sequence[Dict[str, Any]],
    ) -> str:
        body = promote_inline_code_blocks(answer, query=query)
        if not body:
            return body
        code_blocks = list(re.finditer(r"```[\s\S]*?```", body))
        if code_blocks:
            before = body[:code_blocks[0].start()]
            after = body[code_blocks[-1].end():]
            has_intro = self._meaningful_prose_word_count(before) >= 6
            has_outro = self._meaningful_prose_word_count(after) >= 6
        else:
            has_intro = markdown_has_prose_outside_code(body)
            has_outro = True
        intro = self._code_answer_intro(query, related, chunk_hits)
        outro = self._code_answer_outro(query)
        parts: List[str] = []
        if not has_intro:
            parts.append(intro)
        parts.append(body)
        if not has_outro:
            parts.append(outro)
        return "\n\n".join(part.strip() for part in parts if part and part.strip()).strip()

    def _grounded_short_answer(
        self,
        query: str,
        chunk_hits: Sequence[Dict[str, Any]],
        citations: Sequence[Dict[str, Any]],
    ) -> str:
        if not chunk_hits or not citations:
            return ""
        top_item = chunk_hits[0]["item"]
        combined = clean_display_text(
            " ".join(
                part
                for part in [
                    top_item.get("title", ""),
                    top_item.get("text", ""),
                ]
                if clean_display_text(part)
            )
        ).lower()
        query_lower = clean_display_text(query).lower()

        if "computer vision" in query_lower and "high-level understanding" in combined:
            return (
                "In these lectures, computer vision is framed as getting high-level understanding "
                "from digital images and videos."
            )

        if "convolution" in query_lower and "weighted sum" in combined:
            return (
                "In these lectures, convolution is treated as linear filtering: each output pixel "
                "is computed as a weighted sum of a local neighborhood using a kernel."
            )

        if "aperture problem" in query_lower and "ambiguous" in combined:
            return (
                "In these lectures, the aperture problem is the ambiguity of determining motion "
                "when you only observe a small region of an edge."
            )

        if "sift" in query_lower and ("descriptor" in combined or "scale invariant feature transform" in combined):
            return (
                "In these lectures, SIFT is a local feature method that finds keypoints across scale space, "
                "assigns each keypoint an orientation, and represents the surrounding patch with a descriptor for matching."
            )

        if "epipolar" in query_lower and "epipolar line" in combined:
            return (
                "In these lectures, epipolar geometry means that once a point is fixed in one image, "
                "its match in the other image must lie on the corresponding epipolar line."
            )

        return ""

    def _setup_dense_stack(self) -> None:
        if os.getenv("COURSE4186_DISABLE_DENSE", "").strip() == "1":
            return
        if HuggingFaceEmbeddings is None or FAISS is None:
            return
        index_dir = self.artifact_dir / "chunk_index"
        if not ((index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()):
            return
        try:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                encode_kwargs={"normalize_embeddings": True},
            )
            self._vector_store = FAISS.load_local(
                str(index_dir),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            self._dense_enabled = True
        except Exception:
            self._dense_enabled = False
            self._vector_store = None
            self._embeddings = None

    def _setup_llm(self) -> None:
        if OpenAI is None:
            return
        api_key = os.getenv("COURSE_LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("KIMI_API_KEY")
        base_url = os.getenv("COURSE_LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("KIMI_BASE_URL")
        model = os.getenv("COURSE_LLM_MODEL") or os.getenv("DEEPSEEK_CHAT_MODEL") or os.getenv("KIMI_MODEL") or "deepseek-chat"
        if not api_key:
            try:
                from kimi_utils import KIMI_API_BASE, KIMI_API_KEY, KIMI_MODEL

                api_key = KIMI_API_KEY or api_key
                base_url = base_url or KIMI_API_BASE
                model = KIMI_MODEL or model
            except Exception:
                api_key = None
        if not api_key:
            return
        try:
            self._llm_client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
            self._llm_model = model
        except Exception:
            self._llm_client = None
            self._llm_model = None

    @property
    def dense_enabled(self) -> bool:
        return self._dense_enabled

    @property
    def llm_enabled(self) -> bool:
        return self._llm_client is not None and bool(self._llm_model)

    def list_knowledge_points(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for kp in self.kps:
            question_count = len(self.questions_by_kp.get(kp["kp_id"], []))
            if question_count <= 0:
                continue
            rows.append(
                {
                    "kp_id": kp["kp_id"],
                    "name": kp["name"],
                    "description": kp["description"],
                    "weeks": kp.get("weeks", []),
                    "question_count": question_count,
                }
            )
        rows.sort(key=lambda row: (row["weeks"][0] if row["weeks"] else "ZZZ", row["name"]))
        return rows

    def get_kp(self, kp_id: str) -> Optional[Dict[str, Any]]:
        kp = self.kp_by_id.get(kp_id)
        if not kp:
            return None
        questions = list(self.questions_by_kp.get(kp_id, []))
        if not questions:
            return None
        payload = dict(kp)
        payload["questions"] = questions
        return payload

    def related_questions(self, kp_id: str) -> List[Dict[str, Any]]:
        return list(self.questions_by_kp.get(kp_id, []))

    def generate_follow_up_variants(
        self,
        kp_id: str,
        *,
        weak_attempts: Optional[Sequence[Dict[str, Any]]] = None,
        existing_questions: Optional[Sequence[Dict[str, Any]]] = None,
        question_count: int = 1,
    ) -> List[Dict[str, Any]]:
        if not self.llm_enabled or question_count <= 0:
            return []
        kp = self.kp_by_id.get(kp_id)
        if not kp:
            return []

        support_hits = self._follow_up_support_hits(kp, weak_attempts=weak_attempts, limit=4)
        if not support_hits:
            return []

        review_refs = self._dedupe_review_refs(self._build_citations(support_hits, limit=4))
        support_text = "\n\n".join(
            (
                f"Source: {hit['item'].get('relative_path', '')} "
                f"({format_source_location(hit['item'].get('source_type', ''), hit['item'].get('unit_type', ''), hit['item'].get('unit_index', 0))})\n"
                f"Section: {clean_display_text(hit['item'].get('title', '')) or 'Untitled section'}\n"
                f"Excerpt: {safe_snippet(hit['item'].get('text', ''), 360)}"
            )
            for hit in support_hits[:4]
        )
        mistake_text = "\n".join(
            (
                f"- Missed question: {clean_display_text(attempt.get('question', ''))}\n"
                f"  Correct answer: {clean_display_text(attempt.get('reference_answer', ''))}\n"
                f"  Explanation: {clean_display_text(attempt.get('explanation', ''))}"
            )
            for attempt in list(weak_attempts or [])[:2]
            if clean_display_text(attempt.get("question", ""))
        ) or "- No prior mistake details available."
        existing_text = "\n".join(
            f"- {clean_display_text(question.get('question', ''))}"
            for question in list(existing_questions or [])[:6]
            if clean_display_text(question.get("question", ""))
        ) or "- No fixed-bank examples provided."

        prompt = textwrap.dedent(
            f"""
            You are writing follow-up multiple-choice questions for a university computer vision course.
            Use only the supplied course evidence.

            Knowledge point: {kp.get('name', '')}
            Description: {kp.get('description', '')}
            Keywords: {", ".join(kp.get('keywords', []))}

            Student weak-point evidence:
            {mistake_text}

            Existing fixed-bank questions for this topic:
            {existing_text}

            Course evidence:
            {support_text}

            Return strict JSON as a list with {question_count} item(s).
            Each item must contain:
            - question
            - options
            - correct_option
            - explanation

            Requirements:
            - Every item must be a multiple-choice question with exactly 4 options labelled A. to D.
            - There must be exactly one correct answer.
            - Write in lecturer-written quiz style, not chatbot style.
            - Build each question as a fresh but realistic variant of the fixed-bank style shown above.
            - Prefer concrete concept checks, reasoning steps, equation interpretation, method choice, error diagnosis, or small calculation-style prompts when the evidence supports them.
            - Avoid weak prompts such as "Which keyword...", "Which lecture...", "Which topic matches...", "Which knowledge point...", or anything asking about file names, week numbers, or slide numbers.
            - Do not use "All of the above" or "None of the above".
            - Avoid simply paraphrasing the existing fixed-bank questions.
            - If the evidence is too thin to support a concrete lecturer-style question, return an empty list.
            - Keep each question self-contained and answerable from the provided evidence.
            - The explanation must be a short student-facing revision note of 1 or 2 sentences.
            - The explanation must briefly justify the correct answer and connect back to the lecture idea the student needs to review.
            - Do not mention option letters, do not compare wrong options, and do not say "the correct answer is option B".
            - Do not quote slide text, do not mention "lecture excerpt", and do not paste source wording verbatim.
            - Do not invent facts beyond the evidence.
            """
        ).strip()

        try:
            response = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Generate student-facing lecturer-style multiple-choice questions only. Output strict JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.15,
                max_tokens=max(480, question_count * 320),
            )
            parsed = parse_json_response(response.choices[0].message.content or "[]")
        except Exception as exc:
            print(f"[Course4186KnowledgeBase] Follow-up variant generation fallback: {exc}")
            return []

        if isinstance(parsed, dict):
            if {
                "question",
                "options",
                "correct_option",
                "explanation",
            }.issubset(set(parsed.keys())):
                parsed = [parsed]
            else:
                parsed = parsed.get("items") or parsed.get("questions") or []
        if not isinstance(parsed, list):
            return []

        rows: List[Dict[str, Any]] = []
        for index, item in enumerate(parsed, start=1):
            normalized = self._normalize_follow_up_variant(
                kp=kp,
                item=item,
                review_refs=review_refs,
                index=index,
            )
            if normalized:
                rows.append(normalized)
            if len(rows) >= question_count:
                break
        return rows[:question_count]

    def search_knowledge_points(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        ranking_query = self._ranking_query(query)
        if not ranking_query:
            return []
        hits = lexical_rank(
            ranking_query,
            self.kps,
            text_getter=lambda kp: "\n".join([kp["name"], kp["description"], " ".join(kp.get("keywords", []))]),
            top_k=max(top_k * 3, top_k),
        )
        rows = [{"score": round(score, 3), "item": item} for score, item in hits]
        return self._rerank_kp_hits(query, rows, top_k=top_k)

    def search_chunks(
        self,
        query: str,
        top_k: int = 5,
        kp_context: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        ranking_query = self._ranking_query(query)
        if not ranking_query:
            return []
        candidate_limit = max(top_k * 8, 24)
        support_rows = self._support_candidate_rows(kp_context, limit=min(candidate_limit, 16))
        lexical_hits = lexical_rank(
            ranking_query,
            self.chunks,
            text_getter=lambda chunk: "\n".join([chunk["title"], chunk["relative_path"], chunk["text"]]),
            top_k=candidate_limit,
        )
        lexical_rows = [{"score": round(score, 3), "item": item} for score, item in lexical_hits]

        if len(self._content_tokens(query)) <= 3 and (lexical_rows or support_rows):
            return self._rerank_chunk_hits(query, support_rows + lexical_rows, kp_context, top_k=top_k)

        if self._dense_enabled and self._vector_store is not None:
            try:
                dense_hits = self._vector_store.similarity_search_with_score(ranking_query, k=candidate_limit)
                rows: List[Dict[str, Any]] = []
                for document, score in dense_hits:
                    chunk_id = document.metadata.get("chunk_id")
                    chunk = self.chunk_by_id.get(chunk_id)
                    if not chunk:
                        continue
                    rows.append({"score": -float(score), "item": chunk})
                if rows:
                    merged: List[Dict[str, Any]] = []
                    seen_chunk_ids: set[str] = set()
                    for hit in support_rows + rows + lexical_rows:
                        chunk_id = hit["item"].get("chunk_id")
                        if chunk_id in seen_chunk_ids:
                            continue
                        seen_chunk_ids.add(chunk_id)
                        merged.append(hit)
                        if len(merged) >= candidate_limit:
                            break
                    reranked = self._rerank_chunk_hits(query, merged, kp_context, top_k=top_k)
                    if reranked:
                        return reranked
            except Exception:
                pass

        return self._rerank_chunk_hits(query, support_rows + lexical_rows, kp_context, top_k=top_k)

    def answer(self, query: str, history: Optional[List[Dict[str, str]]] = None, top_k: int = 5) -> Dict[str, Any]:
        special_response = self._assistant_response_for_query(query, history or [])
        if special_response:
            return {
                "answer": special_response,
                "citations": [],
                "related_kps": [],
                "mode": "assistant",
            }

        if not self._looks_course_request(query):
            return {
                "answer": self._out_of_scope_response(query, history or []),
                "citations": [],
                "related_kps": [],
                "mode": "assistant",
            }

        retrieval_query = self._contextual_query(query, history or [])
        kp_hits = self.search_knowledge_points(retrieval_query, top_k=3)
        task_profile = self._query_task_profile(retrieval_query, kp_hits)
        chunk_search_k = max(top_k, 8) if task_profile.get("needs_multi_concept_coverage") else top_k
        chunk_hits = self.search_chunks(retrieval_query, top_k=chunk_search_k, kp_context=kp_hits)
        focused_hits = self._focused_kp_support_hits(retrieval_query, kp_hits, limit=6, task_profile=task_profile)
        if task_profile.get("needs_multi_concept_coverage"):
            multi_concept_hits = self._multi_concept_chunk_hits(retrieval_query, chunk_hits, limit=4)
        else:
            multi_concept_hits = []
        if multi_concept_hits:
            citation_hits = multi_concept_hits
        elif task_profile.get("needs_multi_concept_coverage") and len(task_profile.get("relevant_kp_hits") or []) < 2:
            citation_candidates = chunk_hits
            citation_hits = self._preferred_chunk_hits(retrieval_query, citation_candidates, limit=4, kp_context=None)
        else:
            citation_candidates = focused_hits if len(focused_hits) >= 2 else chunk_hits
            citation_hits = self._preferred_chunk_hits(retrieval_query, citation_candidates, limit=4, kp_context=kp_hits)
        citations = self._build_citations(citation_hits, limit=4)
        coverage_level = self._course_coverage_level(retrieval_query, citation_hits)
        related = [
            {
                "kp_id": hit["item"]["kp_id"],
                "name": hit["item"]["name"],
                "description": hit["item"]["description"],
            }
            for hit in kp_hits[:3]
        ]

        if not citations:
            if self._looks_cv_request(query) and self._llm_client is not None and self._llm_model:
                try:
                    llm_payload = self._llm_answer(query, history or [], related, [], [], coverage_level="none", task_profile=task_profile)
                    answer_body = self._strip_uncited_course_connection(llm_payload.get("answer", ""))
                    return {
                        "answer": self._answer_with_source_line(answer_body, []),
                        "citations": [],
                        "related_kps": related,
                        "mode": "llm",
                    }
                except Exception as exc:
                    print(f"[Course4186KnowledgeBase] CV general answer fallback: {exc}")
            return {
                "answer": (
                    "I could not find a clear answer for that in the current Course 4186 materials. "
                    "Try naming the lecture topic, method, or formula more specifically."
                ),
                "citations": [],
                "related_kps": related,
                "mode": "assistant",
            }

        if not self._is_grounded_answer_ready(retrieval_query, kp_hits, citation_hits):
            if self._looks_cv_request(query) and self._llm_client is not None and self._llm_model:
                try:
                    llm_payload = self._llm_answer(
                        query,
                        history or [],
                        related,
                        citation_hits[:4],
                        citations,
                        coverage_level=coverage_level,
                        task_profile=task_profile,
                    )
                    used_sources = llm_payload.get("used_sources", [])
                    if used_sources:
                        final_citations = self._select_final_citations(citations, used_sources, max_count=3)
                    elif task_profile.get("needs_multi_concept_coverage"):
                        final_citations = citations[:3]
                    else:
                        final_citations = []
                    answer_body = llm_payload.get("answer", "")
                    if not final_citations:
                        answer_body = self._strip_uncited_course_connection(answer_body)
                    return {
                        "answer": self._answer_with_source_line(answer_body, final_citations),
                        "citations": final_citations,
                        "related_kps": related,
                        "mode": "llm",
                    }
                except Exception as exc:
                    print(f"[Course4186KnowledgeBase] Partial CV answer fallback: {exc}")
            return {
                "answer": self._insufficient_evidence_response(query, related),
                "citations": [],
                "related_kps": related,
                "mode": "assistant",
            }

        if self._looks_brief_concept_request(query, task_profile=task_profile, kp_hits=kp_hits):
            short_answer = self._grounded_short_answer(query, citation_hits[:3], citations[:3])
            if short_answer:
                final_citations = citations[:2]
                return {
                    "answer": self._answer_with_source_line(short_answer, final_citations),
                    "citations": final_citations,
                    "related_kps": related,
                    "mode": "grounded",
                }

        if self._llm_client is not None and self._llm_model and citations:
            try:
                llm_payload = self._llm_answer(
                    query,
                    history or [],
                    related,
                    citation_hits[:4],
                    citations,
                    coverage_level=coverage_level,
                    task_profile=task_profile,
                )
                used_sources = llm_payload.get("used_sources", [])
                if used_sources:
                    final_citations = self._select_final_citations(citations, used_sources, max_count=3)
                elif task_profile.get("needs_multi_concept_coverage"):
                    final_citations = citations[:3]
                elif coverage_level == "direct":
                    final_citations = citations[:2]
                else:
                    final_citations = []
                answer_body = llm_payload.get("answer", "")
                if not final_citations:
                    answer_body = self._strip_uncited_course_connection(answer_body)
                answer = self._answer_with_source_line(answer_body, final_citations)
                return {
                    "answer": answer,
                    "citations": final_citations,
                    "related_kps": related,
                    "mode": "llm",
                }
            except Exception as exc:
                print(f"[Course4186KnowledgeBase] LLM answer fallback: {exc}")
                pass

        return {
            "answer": self._fallback_answer(query, related, citation_hits[:4], citations, task_profile=task_profile),
            "citations": citations,
            "related_kps": related,
            "mode": "extractive",
        }

    def reference_context(
        self,
        source: str,
        *,
        unit_type: Optional[str] = None,
        unit_index: Optional[int] = None,
        chunk_index: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        normalized_source = normalized_path_key(source)
        if not normalized_source:
            return None
        resolved_unit_type = clean_display_text(unit_type or "").lower()
        resolved_unit_index = int(unit_index or 0)

        chunks: List[Dict[str, Any]] = []
        if resolved_unit_type or resolved_unit_index:
            chunks = list(self.chunks_by_source_unit.get((normalized_source, resolved_unit_type, resolved_unit_index), []))
        if not chunks:
            chunks = [
                chunk for chunk in self.chunks
                if normalized_path_key(chunk.get("relative_path", "")) == normalized_source
            ]
            if resolved_unit_type:
                chunks = [chunk for chunk in chunks if str(chunk.get("unit_type", "")).lower() == resolved_unit_type]
            if resolved_unit_index:
                chunks = [chunk for chunk in chunks if int(chunk.get("unit_index", 0) or 0) == resolved_unit_index]
        if not chunks:
            return None

        chunks.sort(key=lambda row: (int(row.get("unit_index", 0) or 0), int(row.get("chunk_index", 0) or 0)))
        primary = chunks[0]
        if chunk_index:
            for row in chunks:
                if int(row.get("chunk_index", 0) or 0) == int(chunk_index):
                    primary = row
                    break
        excerpt = " ".join(clean_display_text(row.get("text", "")) for row in chunks[:3]).strip()
        location = format_source_location(
            primary.get("source_type", ""),
            primary.get("unit_type", ""),
            primary.get("unit_index", 0),
        )
        return {
            "source": clean_display_text(primary.get("relative_path", "")),
            "display_source": canonical_display_source_name(primary.get("relative_path", "")),
            "source_type": primary.get("source_type", ""),
            "week": clean_display_text(primary.get("week", "")),
            "location": location,
            "unit_type": primary.get("unit_type", ""),
            "unit_index": primary.get("unit_index", 0),
            "section": clean_display_text(primary.get("title", "")) or location,
            "excerpt": excerpt or clean_display_text(primary.get("text", "")),
            "chunks": [
                {
                    "chunk_index": row.get("chunk_index", 0),
                    "text": clean_display_text(row.get("text", "")),
                }
                for row in chunks[:4]
                if clean_display_text(row.get("text", ""))
            ],
        }

    def suggest_session_title(self, user_message: str, assistant_message: str) -> str:
        user_text = clean_display_text(user_message)
        assistant_text = clean_display_text(assistant_message)
        intent = self._assistant_intent(user_text)
        if intent == "identity":
            return "Learning Assistant"
        if intent == "greeting":
            return "Course 4186 Chat"
        if intent == "help":
            return "Using the Assistant"
        if intent == "quiz":
            return "Quiz Practice"
        if intent == "report":
            return "Learning Report"
        if intent == "thanks":
            return "Course 4186 Chat"
        if self._llm_client is not None and self._llm_model:
            try:
                prompt = textwrap.dedent(
                    f"""
                    Create a very short conversation title for a course chat session.
                    Use the same language as the conversation.
                    Requirements:
                    - 2 to 6 words
                    - plain student-facing wording
                    - summarize the actual topic, not the wording of the question
                    - no quotation marks
                    - no trailing punctuation

                    Student:
                    {user_text}

                    Assistant:
                    {assistant_text[:500]}
                    """
                ).strip()
                response = self._llm_client.chat.completions.create(
                    model=self._llm_model,
                    messages=[
                        {"role": "system", "content": "You write concise chat titles."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=24,
                )
                title = clean_display_text(response.choices[0].message.content or "")
                title = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", title).strip()
                title = re.sub(r"[.!?。！？]+$", "", title).strip()
                if title:
                    return title[:56]
            except Exception:
                pass

        tokens = [
            token for token in tokenize(user_text)
            if len(token) > 2 and token not in STOPWORDS and token not in {"what", "explain", "about", "course", "please", "help", "introduce", "sth", "something"}
        ]
        if "computer" in tokens and "vision" in tokens:
            return "Computer Vision"
        if "epipolar" in tokens:
            return "Epipolar Geometry"
        if "sift" in tokens:
            return "SIFT Features"
        if "convolution" in tokens:
            return "Convolution Basics"
        if "edge" in tokens and "detection" in tokens:
            return "Edge Detection"
        if tokens:
            return " ".join(token.capitalize() for token in tokens[:4])
        return "Course 4186 Chat"

    def _select_final_citations(
        self,
        citations: Sequence[Dict[str, Any]],
        used_source_ids: Sequence[str],
        max_count: int = 3,
    ) -> List[Dict[str, Any]]:
        by_id = {
            clean_display_text(citation.get("citation_id", "")).upper(): citation
            for citation in citations
            if clean_display_text(citation.get("citation_id", ""))
        }
        selected: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for source_id in used_source_ids:
            resolved_id = clean_display_text(source_id).upper()
            citation = by_id.get(resolved_id)
            if not citation or resolved_id in seen:
                continue
            seen.add(resolved_id)
            selected.append(citation)
            if len(selected) >= max_count:
                return selected
        for citation in citations:
            resolved_id = clean_display_text(citation.get("citation_id", "")).upper()
            if resolved_id in seen:
                continue
            seen.add(resolved_id)
            selected.append(citation)
            if len(selected) >= max_count:
                break
        return selected[:max_count]

    def _course_source_line(self, citations: Sequence[Dict[str, Any]]) -> str:
        if not citations:
            return ""
        refs = [student_facing_source_reference(citation) for citation in citations[:3]]
        return "Course source: " + "; ".join(refs) + "."

    def _answer_with_source_line(self, body: str, citations: Sequence[Dict[str, Any]]) -> str:
        cleaned = sanitize_transport_answer_text(extract_student_answer_text(body))
        cleaned = strip_course_source_line(cleaned)
        cleaned = re.sub(r"(?:\n|\r|\s)*used_sources\s*:.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        cleaned = strip_citation_markers(cleaned)
        cleaned = normalize_answer_body_sources(cleaned)
        return rebuild_answer_with_citations(cleaned, citations)

    def _strip_uncited_course_connection(self, text: str) -> str:
        cleaned = normalize_answer_body_sources(extract_student_answer_text(text))
        if "```" in cleaned:
            return cleaned
        sentences = re.split(r"(?<=[.!?])\s+", cleaned.strip())
        if not sentences:
            return cleaned
        stripped: List[str] = []
        course_openers = (
            "in this course", "in our course", "in the course", "in the context of this course",
            "in the context of our course", "in our lectures", "in the lectures",
            "in lecture", "in week", "our course", "the course",
        )
        course_markers = (
            "lecture materials", "the lectures", "our lectures", "this course", "our course",
            ".pdf", "page ",
        )
        for sentence in sentences:
            lowered = sentence.strip().lower()
            if any(lowered.startswith(prefix) for prefix in course_openers):
                continue
            if any(marker in lowered for marker in course_markers):
                continue
            stripped.append(sentence.strip())
        return " ".join(part for part in stripped if part).strip() or cleaned.strip()

    def _llm_answer(
        self,
        query: str,
        history: List[Dict[str, str]],
        related: List[Dict[str, Any]],
        chunk_hits: Sequence[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        coverage_level: str = "direct",
        task_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        code_request = self._query_requests_code(query)
        task_profile = task_profile or self._query_task_profile(query)
        repeat_context = self._repeat_answer_context(query, history)
        repeat_count = int(repeat_context.get("repeat_count", 0) or 0)
        recent_similar_answers = list(repeat_context.get("recent_answers", []))
        recent_openings = list(repeat_context.get("recent_openings", []))
        history_text = "\n".join(
            f"{turn.get('role', 'user')}: {turn.get('content', '').strip()}"
            for turn in history[-6:]
            if turn.get("content")
        )
        evidence = "\n\n".join(
            (
                f"[{citation['citation_id']}] {citation.get('display_source') or citation['source']} "
                f"({citation.get('location') or ('chunk ' + str(citation['chunk_index']))})\n"
                f"Section: {citation.get('section') or 'Untitled section'}\n"
                f"Excerpt: {safe_snippet(hit['item'].get('text', ''), 520)}"
            )
            for hit, citation in zip(chunk_hits, citations)
        )
        kp_text = "\n".join(f"- {item['name']}" for item in related) or "- None"
        recent_answer_block = "\n".join(
            f"- {safe_snippet(answer, 220)}"
            for answer in recent_similar_answers
        ) or "- None"
        recent_opening_block = "\n".join(
            f"- {safe_snippet(opening, 140)}"
            for opening in recent_openings
        ) or "- None"
        relevant_topics = "\n".join(
            f"- {clean_display_text(hit['item'].get('name', ''))}"
            for hit in (task_profile.get("relevant_kp_hits") or [])
            if clean_display_text(hit["item"].get("name", ""))
        ) or "- None"
        repeat_guidance = (
            f"This question, or a nearly identical one, already appeared {repeat_count} time(s) earlier in this conversation. "
            "Keep the facts consistent, but do not reuse the same opening sentence, explanation order, or sentence rhythm from the recent similar answers. "
            f"Use this variation angle for the new answer: {repeat_context.get('style_instruction', '')}"
            if repeat_count > 0
            else "Use the clearest teaching-assistant explanation for a first answer to this question."
        )
        task_guidance = self._task_prompt_guidance(task_profile)
        prompt = textwrap.dedent(
            f"""
            You are the Course 4186 student learning assistant for a computer vision course.
            Answer naturally, in the same language as the student's question.
            The student may ask a standard computer vision question even when the exact term is not explicitly defined in the course slides.
            In that case, you may answer from reliable computer vision knowledge first, then connect it to the course only if the supplied evidence is genuinely relevant.
            Never invent course coverage, and never turn a loosely related lecture mention into a claimed lecture definition.
            Do not open with defensive phrases such as "the term is not directly covered in the materials" or "the provided lecture materials do not define it".
            If the course evidence is partial, give the concept explanation first, then add one brief sentence about the closest lecture connection.
            If the course evidence is weak or unrelated, answer the concept normally and leave course sources unused.
            Do not paste raw chunk labels or say generic phrases like "I found grounded material".
            For direct lecture-covered topics, prefer the lecture's framing over an encyclopedia-style introduction.
            For broader CV questions, start with a clean student-facing definition, then briefly mention the lecture angle if available.
            Use the lecture evidence to connect the concept to this course instead of quoting slide fragments.
            Write like a teaching assistant helping a student revise this lecture before a quiz.
            Keep the answer concise, natural, and specific to the cited lecture material.
            Use 2 or 3 sentences unless the student explicitly asks for a longer derivation or comparison.
            Avoid filler phrases, repeated restatement, and broad background context that is not visible in the evidence.
            If this question, or a very similar one, appeared earlier in the same conversation, you must answer it in clearly different wording and structure while preserving the facts.
            Never reuse the same first sentence as a recent similar answer.
            Avoid stock textbook phrases such as "fundamental concept", "various tasks", or "in the field of computer vision" unless the lecture evidence itself uses that framing.
            Avoid generic openings such as "X is a field of study", "X focuses on", "X involves techniques", or "X is a popular method" unless those exact ideas are directly supported by the evidence.
            Do not introduce course-specific claims, causes, applications, or mechanisms unless they are explicitly supported by the evidence.
            If the evidence states only the core idea or one example, keep the lecture connection at that level and do not over-expand it.
            Do not mention lecture file names, page numbers, slide numbers, or source ids in the answer body.
            Do not include evidence labels such as [S1], [S2], or any source-id notation in the answer body.
            Do not include a source list line inside the answer body.
            Course coverage level for this question: {coverage_level}.
            If coverage level is "direct", make the lecture connection central.
            If coverage level is "related", make the concept explanation central and the lecture connection secondary.
            If coverage level is "none", do not claim the lectures covered the concept.
            If you include any explicit sentence about this course or its lectures, you must select at least one matching source id in used_sources.
            If no trustworthy source fits, omit the course-connection sentence completely.
            Task guidance:
            {task_guidance}
            Variation requirement:
            {repeat_guidance}
            {"If the student asks for code, include one short explanatory sentence before the code block and one short practical note after it. Include the code as a fenced Markdown code block with its own lines and the correct language label such as ```python. Never place multi-line code inside single backticks, and do not return code only." if code_request else ""}
            Any course-specific factual point in the answer must be supported by the selected evidence ids.
            When no trustworthy course source is needed, return an empty used_sources array.
            Return strict JSON with:
            - answer: the student-facing explanation only
            - used_sources: an array of 0 to 3 ids chosen only from the evidence labels such as ["S1", "S2"]

            Conversation:
            {history_text or 'None'}

            Recent similar-answer openings to avoid reusing:
            {recent_opening_block}

            Recent answers to similar questions:
            {recent_answer_block}

            User question:
            {query}

            Relevant course topics inferred from the question:
            {relevant_topics}

            Matched knowledge points:
            {kp_text}

            Evidence:
            {evidence or 'None'}
            """
        ).strip()
        temperature = 0.18 if repeat_count <= 0 else min(0.42, 0.18 + 0.07 * repeat_count)

        def request_raw_content(*, extra_instruction: str = "", temperature_override: Optional[float] = None) -> str:
            prompt_body = prompt
            if extra_instruction.strip():
                prompt_body = (
                    f"{prompt}\n\nAdditional requirement for this attempt:\n"
                    f"{extra_instruction.strip()}"
                )
            response = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise teaching assistant for a computer vision course. "
                            "Use supplied course evidence when it genuinely applies, but you may answer standard computer vision questions from reliable domain knowledge. "
                            "Never invent lecture coverage or fake citations. "
                            "When the student repeats a question in the same chat, keep the facts stable but vary the phrasing and structure."
                        ),
                    },
                    {"role": "user", "content": prompt_body},
                ],
                temperature=temperature if temperature_override is None else temperature_override,
                max_tokens=420 if code_request else 220,
            )
            return (response.choices[0].message.content or "").strip()

        def parse_answer_payload(raw_content: str) -> Dict[str, Any]:
            raw_used_sources = extract_used_source_ids(raw_content)
            try:
                parsed = parse_json_response(raw_content)
                if isinstance(parsed, dict):
                    answer_text = str(parsed.get("answer") or "").strip()
                    used_sources_raw = parsed.get("used_sources", [])
                    used_sources = [
                        clean_display_text(item).upper()
                        for item in used_sources_raw
                        if clean_display_text(item)
                    ] if isinstance(used_sources_raw, list) else []
                    if not used_sources:
                        used_sources = raw_used_sources
                    if answer_text:
                        answer_text = (
                            self._ensure_code_answer_explanation(answer_text, query, related, chunk_hits)
                            if code_request
                            else sanitize_transport_answer_text(answer_text)
                        )
                        if contains_transport_artifacts(answer_text):
                            raise ValueError("LLM answer still contained transport artifacts after JSON parse.")
                        final_answer = self._trim_optional_course_tail(answer_text, task_profile)
                        return {
                            "answer": extract_student_answer_text(final_answer),
                            "used_sources": used_sources,
                        }
            except Exception:
                pass

            fallback_answer = sanitize_transport_answer_text(extract_student_answer_text(raw_content))
            if not fallback_answer:
                raise ValueError("LLM answer did not contain a usable student-facing answer.")
            if code_request:
                fallback_answer = self._ensure_code_answer_explanation(fallback_answer, query, related, chunk_hits)
            if contains_transport_artifacts(fallback_answer):
                repaired = sanitize_transport_answer_text(
                    normalize_answer_body_sources(
                        strip_course_source_line(
                            strip_source_id_list_suffix(fallback_answer)
                        )
                    )
                )
                if repaired:
                    fallback_answer = repaired
            if contains_transport_artifacts(fallback_answer):
                fallback_answer = sanitize_transport_answer_text(strip_source_id_list_suffix(raw_content))
            if not fallback_answer:
                raise ValueError("LLM answer fallback could not be repaired.")
            fallback_answer = self._trim_optional_course_tail(fallback_answer, task_profile)
            return {
                "answer": fallback_answer,
                "used_sources": raw_used_sources,
            }

        payload = parse_answer_payload(request_raw_content())
        if not self._answer_satisfies_task(payload.get("answer", ""), query, task_profile):
            retry_instruction = (
                "The current draft does not fully satisfy the task. "
                f"{task_guidance} "
                "Make sure every named concept in the question is addressed explicitly before you finish."
            )
            payload = parse_answer_payload(
                request_raw_content(
                    extra_instruction=retry_instruction,
                    temperature_override=min(0.46, temperature + 0.08),
                )
            )
        if repeat_count > 0 and self._answer_too_similar_to_recent(payload.get("answer", ""), recent_similar_answers):
            retry_instruction = (
                "You are answering a repeated question in the same chat. "
                "Rewrite the explanation with a clearly different opening sentence and a different explanation path from every recent similar answer above. "
                "Do not start with any of the listed recent openings. "
                "If the recent answer defined the term first, switch to intuition, geometry, computation, or exam-revision framing instead."
            )
            retry_payload = parse_answer_payload(
                request_raw_content(
                    extra_instruction=retry_instruction,
                    temperature_override=min(0.48, temperature + 0.12),
                )
            )
            payload = retry_payload
        if not self._answer_satisfies_task(payload.get("answer", ""), query, task_profile):
            raise ValueError("LLM answer did not satisfy the task requirements.")
        return payload

    def _llm_assistant_response(self, query: str, intent: str, history: Sequence[Dict[str, str]]) -> str:
        intent_guidance = {
            "greeting": "Greet the student naturally and invite a course-related question.",
            "identity": "Explain who you are in one or two plain, professional sentences.",
            "help": "Explain briefly what kinds of Course 4186 help you can provide.",
            "report": "Explain briefly what students can see in the Learning Report page.",
            "quiz": "Tell the student how to use quiz practice in a natural way.",
            "thanks": "Reply politely and invite the next course-related question.",
            "out_of_scope": "Politely but clearly state that the question is outside Course 4186, so you will not answer it directly. This explicit refusal is required. After that, redirect the student to nearby course topics in a natural way. If the student repeats another unrelated question, vary the wording and avoid repeating the same refusal sentence.",
        }
        recent_assistant_replies = [
            clean_display_text(turn.get("content", ""))
            for turn in history[-8:]
            if str(turn.get("role") or "").lower() == "assistant" and clean_display_text(turn.get("content", ""))
        ]
        recent_reply_block = "\n".join(f"- {item}" for item in recent_assistant_replies[-3:]) or "- None"
        prompt = textwrap.dedent(
            f"""
            You are the Course 4186 student learning assistant.
            Reply in the same language as the student's message.
            Keep the reply natural, short, student-facing, and professionally neutral.
            Prefer one or two concise sentences.
            Avoid cheerleading, jokes, emojis, and overly enthusiastic phrases.
            Do not mention internal rules, retrieval, grounding, or system behavior.
            Do not answer unrelated factual questions outside Course 4186.
            Avoid repeating the same wording, opening phrase, or sentence structure used in the recent assistant replies.
            If the student asks the same assistant-type question again, vary the phrasing while keeping the meaning consistent.
            For out-of-scope questions, explicitly say that the question is outside Course 4186 and that you will not answer it directly. Do not merely redirect without stating the refusal.

            Interaction type:
            {intent}

            Guidance:
            {intent_guidance.get(intent, "Reply naturally and stay within the Course 4186 assistant role.")}

            Recent assistant replies:
            {recent_reply_block}

            Student message:
            {clean_display_text(query)}
            """
        ).strip()
        response = self._llm_client.chat.completions.create(
            model=self._llm_model,
            messages=[
                {"role": "system", "content": "You are a concise course assistant. For out-of-scope questions, you must explicitly refuse to answer the unrelated question before redirecting."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=180,
        )
        return clean_display_text(response.choices[0].message.content or "")

    def _fallback_answer(
        self,
        query: str,
        related: List[Dict[str, Any]],
        chunk_hits: Sequence[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        task_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        profile = task_profile or self._query_task_profile(query)
        if not citations:
            return (
                "I could not find a clear answer for that question in the current Course 4186 materials. "
                "Try using a more specific topic such as edge detection, SIFT, stereo vision, epipolar geometry, or optical flow."
            )

        if profile.get("is_comparison") and len(related) >= 2:
            def display_label(item: Dict[str, Any]) -> str:
                name = clean_display_text(item.get("name", ""))
                lowered = name.lower()
                if "structure from motion" in lowered:
                    return "Structure from motion"
                if "convolution" in lowered:
                    return "Convolution"
                if "epipolar geometry" in lowered:
                    return "Epipolar geometry"
                if "homography" in lowered:
                    return "Homography"
                return name or "This topic"

            relevant_hits = list(profile.get("relevant_kp_hits") or [])
            ordered_related = [
                {
                    "kp_id": hit["item"]["kp_id"],
                    "name": hit["item"]["name"],
                    "description": hit["item"]["description"],
                }
                for hit in relevant_hits[:2]
            ]
            seen_related_ids = {item["kp_id"] for item in ordered_related}
            for item in related:
                if item["kp_id"] in seen_related_ids:
                    continue
                ordered_related.append(item)
                seen_related_ids.add(item["kp_id"])
                if len(ordered_related) >= 2:
                    break
            if len(ordered_related) < 2:
                return (
                    "I found one of the two topics clearly in the current lecture materials, but I could not recover a reliable second course topic for a clean comparison. "
                    "Please try naming the second method more specifically."
                )
            first = ordered_related[0]
            second = ordered_related[1]
            first_desc = first["description"].rstrip(".")
            second_desc = second["description"].rstrip(".")
            parts = [
                f"{display_label(first)} is about {first_desc.lower()}, whereas {display_label(second)} is about {second_desc.lower()}.",
                "So one is a vision problem or reconstruction task, while the other is a local filtering operation applied to image neighborhoods."
                if (
                    "structure from motion" in first["name"].lower()
                    or "structure from motion" in second["name"].lower()
                    or "convolution" in first["name"].lower()
                    or "convolution" in second["name"].lower()
                )
                else "So they play different roles in the course rather than describing the same kind of method.",
                "Course source: " + "; ".join(student_facing_source_reference(item) for item in citations[:3]) + ".",
            ]
            return " ".join(part.strip() for part in parts if part.strip())

        top_kp = related[0] if related else None
        top_hit = chunk_hits[0]["item"] if chunk_hits else {}
        section = clean_display_text(top_hit.get("title", "")) or citations[0].get("section", "the matched lecture section")
        source_ref = student_facing_source_reference(citations[0])
        top_excerpt = safe_snippet(top_hit.get("text", "") or citations[0].get("snippet", ""), 260)
        parts: List[str] = []

        if top_kp:
            if "use" in query.lower() or "used for" in query.lower() or "why" in query.lower():
                parts.append(f"{top_kp['name']} in this course is mainly used for {top_kp['description'].rstrip('.').lower()}.")
            else:
                parts.append(f"{top_kp['name']} in Course 4186 is about {top_kp['description'].rstrip('.').lower()}.")
        else:
            parts.append("The closest matched course material points to the following idea.")

        if section and top_excerpt:
            parts.append(f"The clearest supporting section is {section} in {source_ref}. It highlights: {top_excerpt}.")
        elif top_excerpt:
            parts.append(f"The closest lecture evidence is from {source_ref}. It highlights: {top_excerpt}.")

        extra_sources = [student_facing_source_reference(item) for item in citations[1:3]]
        source_line = [source_ref] + extra_sources
        parts.append("Course source: " + "; ".join(source_line) + ".")
        return " ".join(part.strip() for part in parts if part.strip())

    def recommendation_for_weak_points(self, weak_points: Sequence[Dict[str, Any]]) -> List[str]:
        suggestions: List[str] = []
        for row in weak_points[:3]:
            kp = self.kp_by_id.get(row["kp_id"])
            if not kp:
                continue
            weeks = ", ".join(kp.get("weeks", [])[:2])
            suggestions.append(
                f"Review {kp['name']} first. It appears in {weeks or 'the course materials'}, and your recent accuracy there is {row['accuracy']}%."
            )
        if not suggestions:
            suggestions.append("No weak area is obvious yet. Start with one foundational topic such as filtering, edge detection, or camera geometry.")
        return suggestions
