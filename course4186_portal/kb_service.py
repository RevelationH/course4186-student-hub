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

# Keep embedding retrieval on the PyTorch path so the portal does not inherit
# TensorFlow/Keras conflicts from the shared base environment.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
PRACTICE_OPTION_LINE_RE = re.compile(r"^\s*(?:[-*]\s*)?([A-D])(?:[.):]|\))\s*(.+?)\s*$", re.IGNORECASE)
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
    "my", "our", "ours", "yourself", "introduce", "today", "tell", "regarding",
    "concerning",
}
QUERY_INTENT_TOKENS = {
    "what", "who", "why", "how", "when", "where", "which", "tell", "show",
    "give", "explain", "describe", "define", "introduce", "overview",
    "about", "example", "examples", "sample", "samples", "code",
    "write", "build", "building", "compute", "computing", "implementation",
    "implement", "python", "pytorch", "torch", "simple",
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
    "neural rendering",
    "neural radiance field",
    "radiance field",
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
    "yolo", "bbox", "bounding", "box", "boxes", "nerf", "radiance", "rendering",
    "gaussian", "gaussians", "slam", "voxel", "voxels",
}
GREETING_RE = re.compile(r"^\s*(hi|hello|hey|good morning|good afternoon|good evening)\s*[!.?]*\s*$", re.IGNORECASE)
IDENTITY_RE = re.compile(r"^\s*(who are you|what are you|introduce yourself)\s*[!.?]*\s*$", re.IGNORECASE)
HELP_RE = re.compile(
    r"\b("
    r"what can you do|"
    r"what can you help me with|"
    r"what can i ask you here|"
    r"how can you help|"
    r"how can you help with this course|"
    r"help me use this system"
    r")\b",
    re.IGNORECASE,
)
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
QUESTION_GENERATION_RE = re.compile(
    r"\b("
    r"(give|make|create|write|design|generate|set|prepare|ask)(?:\s+me)?\s+(?:an?\s+|one\s+|a\s+)?"
    r"(question|questions|problem|problems|exercise|exercises|quiz|quizzes)|"
    r"practice\s+(question|questions|problem|problems|exercise|exercises)|"
    r"quiz\s+(question|questions)|"
    r"test\s+me|"
    r"ask\s+me\s+(?:a|an|one)?\s*(question|problem|exercise)"
    r")\b",
    re.IGNORECASE,
)
MULTIPLE_CHOICE_REQUEST_RE = re.compile(
    r"\b(multiple choice|multiple-choice|mcq|options?|a\.\s|b\.\s|c\.\s|d\.\s)\b",
    re.IGNORECASE,
)
CALCULATION_QUESTION_RE = re.compile(
    r"\b(calculate|calculation|compute|computation|work out|numerical|derive)\b",
    re.IGNORECASE,
)
CODE_REQUEST_RE = re.compile(
    r"\b("
    r"code|python|pytorch|snippet|sample code|example code|implementation|implement|write code|show code|demo code|"
    r"program|script|conv1d|conv2d|torch\.nn|torch\.nn\.functional"
    r")\b",
    re.IGNORECASE,
)
FOLLOWUP_REFERENCE_RE = re.compile(
    r"\b("
    r"it|that|this|these|those|them|one|ones|again|same|above|before|previous|earlier"
    r")\b",
    re.IGNORECASE,
)
FOLLOWUP_ACTION_RE = re.compile(
    r"^\s*(?:and|also|then|so|now|ok|okay|well)?\s*("
    r"show|give|write|implement|explain|compare|contrast|derive|expand|elaborate|"
    r"summarize|use|apply|why|how|what about|how about|can you|could you|would you|"
    r"tell me more|go on|continue"
    r")\b",
    re.IGNORECASE,
)
REFERENCE_FOLLOWUP_RE = re.compile(
    r"\b("
    r"review|revise|open|lecture|lectures|source|sources|pdf|page|pages|"
    r"material|materials|section|sections|relevant"
    r")\b",
    re.IGNORECASE,
)
COURSE_QUERY_ALIAS_RULES = (
    (re.compile(r"\bsfm\b", re.IGNORECASE), "structure from motion"),
    (re.compile(r"\bstructure[- ]from[- ]motion\b", re.IGNORECASE), "structure from motion"),
)
SAFE_CV_ALIAS_RE = re.compile(r"\bcv\b", re.IGNORECASE)
CV_RESUME_CONTEXT_TOKENS = {
    "resume", "curriculum", "vitae", "job", "jobs", "application", "applications",
    "cover", "letter", "linkedin", "write", "improve", "submit", "template",
}
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
FOLLOWUP_NON_TOPIC_TOKENS = {
    "code", "snippet", "example", "examples", "implementation", "implement",
    "c++", "cpp", "python", "pytorch", "torch", "java", "javascript", "js",
    "illustrate", "illustration", "demo", "demonstrate", "walkthrough",
    "question", "questions", "problem", "problems", "exercise", "exercises",
    "quiz", "quizzes", "practice", "answer", "answers", "solution", "solutions",
    "step", "steps", "detail", "details", "more", "further",
    "topic", "topics", "related", "cover", "covered", "directly", "first",
    "review", "revise", "open", "lecture", "lectures", "source", "sources",
    "pdf", "page", "pages", "material", "materials", "section", "sections",
    "relevant",
}
STRUCTURAL_QUERY_TOKENS = {
    "compare", "comparison", "different", "difference", "differences", "distinguish",
    "contrast", "versus", "versu", "vs", "between", "relation", "relationship",
    "related", "connection", "connected", "link", "links", "linked", "together",
    "works", "work", "fit", "fits", "similar", "similarity", "similarities",
    "example", "examples", "sample", "samples", "show", "give", "tell", "about",
    "explain", "describe", "define", "definition", "overview", "used", "use",
    "application", "applications", "comparing", "compared", "using",
    "question", "questions", "problem", "problems", "exercise", "exercises",
    "quiz", "quizzes", "practice", "multiple", "choice", "choices", "mcq",
    "option", "options", "select", "best", "correct", "wrong", "answer",
    "topic", "topics", "cover", "covered", "directly", "review", "revise",
    "open", "lecture", "lectures", "source", "sources", "pdf", "page", "pages",
    "material", "materials", "section", "sections", "relevant", "first",
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
    if "\n" not in candidate:
        simple_assignment_pattern = re.compile(
            r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*(?:\([^()\n]+\)|\[[^\[\]\n]+\]|\{[^{}\n]+\}|[-+]?[0-9.]+)\s*$"
        )
        if simple_assignment_pattern.match(candidate):
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
    if "\n" not in candidate:
        strong_single_line_signals = [
            r"\bimport\b",
            r"\bfrom\b",
            r"\bdef\b",
            r"\bclass\b",
            r"\breturn\b",
            r"\bprint\s*\(",
            r"\btorch\.",
            r"\bnn\.",
            r"\bconv[12]d\b",
        ]
        strong_match_count = sum(1 for pattern in strong_single_line_signals if re.search(pattern, candidate))
        return strong_match_count >= 1 and matched >= 4
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


def balance_markdown_code_fences(text: str) -> str:
    candidate = str(text or "").rstrip()
    if not candidate:
        return ""
    if candidate.count("```") % 2 == 0:
        return candidate
    return f"{candidate}\n```"


def normalize_fenced_code_layout(text: str) -> str:
    candidate = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not candidate or "```" not in candidate:
        return candidate
    parts = re.split(r"(```[A-Za-z0-9_-]*\n[\s\S]*?\n```)", candidate)
    rebuilt: List[str] = []
    for part in parts:
        if not part:
            continue
        if part.startswith("```"):
            if rebuilt:
                rebuilt[-1] = rebuilt[-1].rstrip() + "\n\n"
            rebuilt.append(part.strip())
            rebuilt.append("\n\n")
            continue
        rebuilt.append(part.strip())
    result = "".join(rebuilt)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result


def normalize_code_fence_boundaries(text: str) -> str:
    candidate = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not candidate or "```" not in candidate:
        return candidate.strip()
    candidate = re.sub(r"([^\n])(```[A-Za-z0-9_-]*\n)", r"\1\n\n\2", candidate)
    candidate = re.sub(r"(```[A-Za-z0-9_-]*\n[\s\S]*?```)(?=[^\n])", r"\1\n\n", candidate)
    candidate = re.sub(r"\n{3,}", "\n\n", candidate)
    return candidate.strip()


def _cleanup_latex_token(token: str) -> str:
    cleaned = str(token or "")
    replacements = (
        (r"\\mathbf\{([^{}]+)\}", r"\1"),
        (r"\\vec\{([^{}]+)\}", r"\1"),
        (r"\\hat\{([^{}]+)\}", r"\1-hat"),
        (r"\\times", " x "),
        (r"\\cdot", " · "),
        (r"\\left", ""),
        (r"\\right", ""),
        (r"\\,", " "),
        (r"\\;", " "),
        (r"\\!", ""),
    )
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned)
    cleaned = cleaned.replace("{", "").replace("}", "")
    cleaned = cleaned.replace("\\", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _normalize_latex_non_matrix_segment(text: str) -> str:
    cleaned = str(text or "")
    replacements = (
        (r"\\mathbf\{([^{}]+)\}", r"\1"),
        (r"\\vec\{([^{}]+)\}", r"\1"),
        (r"\\hat\{([^{}]+)\}", r"\1-hat"),
        (r"\\times", " x "),
        (r"\\cdot", " · "),
        (r"\\left", ""),
        (r"\\right", ""),
        (r"\\,", " "),
        (r"\\;", " "),
        (r"\\!", ""),
    )
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned)
    cleaned = cleaned.replace("{", "").replace("}", "")
    cleaned = cleaned.replace("\\(", "").replace("\\)", "")
    cleaned = cleaned.replace("\\[", "\n\n").replace("\\]", "\n\n")
    cleaned = cleaned.replace("\\", "")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"(?m)^\s*[\[\]]\s*$", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*&+\s*$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _format_latex_matrix_block(content: str, *, determinant: bool) -> str:
    rows: List[List[str]] = []
    for raw_row in re.split(r"\\\\", str(content or "")):
        row = raw_row.strip()
        if not row:
            continue
        cells = [_cleanup_latex_token(cell) for cell in row.split("&")]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    if not determinant and all(len(row) == 1 for row in rows):
        return "(" + ", ".join(row[0] for row in rows) + ")"
    if not determinant and len(rows) == 1:
        return "(" + ", ".join(rows[0]) + ")"

    if determinant:
        ascii_rows = ["| " + "  ".join(row) + " |" for row in rows]
    else:
        ascii_rows = ["[ " + "  ".join(row) + " ]" for row in rows]
    return "\n\n```text\n" + "\n".join(ascii_rows) + "\n```\n\n"


def normalize_chat_math_notation(text: str) -> str:
    candidate = str(text or "")
    if not candidate:
        return ""

    parts = re.split(r"(```[\s\S]*?```)", candidate)
    normalized_parts: List[str] = []
    for part in parts:
        if not part:
            continue
        if part.startswith("```"):
            normalized_parts.append(part)
            continue

        segment = part
        segment = re.sub(
            r"=\s*\\begin\{vmatrix\}([\s\S]*?)\\end\{vmatrix\}\s*=\s*([^\n]+)",
            lambda match: (
                " =\n"
                + _format_latex_matrix_block(match.group(1), determinant=True)
                + "which expands to "
                + _normalize_latex_non_matrix_segment(match.group(2))
            ),
            segment,
        )
        segment = re.sub(
            r"\\begin\{pmatrix\}([\s\S]*?)\\end\{pmatrix\}",
            lambda match: _format_latex_matrix_block(match.group(1), determinant=False),
            segment,
        )
        segment = re.sub(
            r"\\begin\{vmatrix\}([\s\S]*?)\\end\{vmatrix\}",
            lambda match: _format_latex_matrix_block(match.group(1), determinant=True),
            segment,
        )
        segment = _normalize_latex_non_matrix_segment(segment)
        normalized_parts.append(segment)

    rebuilt = "".join(normalized_parts)
    rebuilt = re.sub(r"\n{3,}", "\n\n", rebuilt).strip()
    return rebuilt


def normalize_inline_math_operators(text: str) -> str:
    candidate = str(text or "")
    if not candidate:
        return ""

    parts = re.split(r"(```[\s\S]*?```|`[^`\n]+`)", candidate)
    normalized_parts: List[str] = []
    multiplication_pattern = re.compile(r"(?<=[0-9A-Za-z\)\]])\s*\*\s*(?=[0-9A-Za-z\(\[])", re.UNICODE)

    for part in parts:
        if not part:
            continue
        if part.startswith("```") or (part.startswith("`") and part.endswith("`")):
            normalized_parts.append(part)
            continue
        segment = multiplication_pattern.sub(" x ", part)
        segment = re.sub(r"[ \t]{2,}", " ", segment)
        normalized_parts.append(segment)

    rebuilt = "".join(normalized_parts)
    rebuilt = re.sub(r" *\n *", "\n", rebuilt)
    rebuilt = re.sub(r"\n{3,}", "\n\n", rebuilt)
    return rebuilt.strip()


def prefers_plain_numeric_math_example(query: str) -> bool:
    lowered = clean_display_text(query).lower()
    asks_example = bool(re.search(r"\b(example|for example|show me|give me)\b", lowered))
    math_topics = (
        "cross product",
        "dot product",
        "vector",
        "vectors",
        "matrix",
        "matrices",
        "determinant",
        "eigenvalue",
        "eigenvector",
        "homography",
        "epipolar",
        "convolution",
    )
    return asks_example and any(topic in lowered for topic in math_topics)


def answer_uses_unfriendly_math_format(answer: str) -> bool:
    candidate = str(answer or "")
    if not candidate:
        return False
    contains_latex_commands = bool(
        re.search(r"\\(?:begin|end|vec|hat|times|cdot|left|right)\b", candidate)
    )
    contains_latex_delimiters = any(token in candidate for token in ("\\(", "\\)", "\\[", "\\]"))
    has_broken_display_block = bool(
        re.search(r"(?m)^\s*[\[\]]\s*$", candidate)
        and (contains_latex_commands or "&" in candidate or "_" in candidate)
    )
    has_matrix_row = bool(re.search(r"(?m)^\s*\|\s*(?:[-0-9A-Za-z_+*/().]+\s+){1,}[-0-9A-Za-z_+*/().]+\s*\|\s*$", candidate))
    has_ampersand_row = bool(re.search(r"(?m)^\s*[^`\n]*&[^`\n]*&[^`\n]*$", candidate))
    return bool(
        "```" in candidate
        or contains_latex_commands
        or contains_latex_delimiters
        or has_broken_display_block
        or has_matrix_row
        or has_ampersand_row
    )


def strip_unfriendly_math_tail(text: str) -> str:
    candidate = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not candidate:
        return ""

    tail_patterns = (
        re.compile(r"(?m)^\s*[\[\]]\s*$"),
        re.compile(r"\\begin\{(?:v|p|b)matrix\}"),
        re.compile(r"(?m)^\s*[^`\n]*&[^`\n]*&[^`\n]*$"),
    )
    for pattern in tail_patterns:
        match = pattern.search(candidate)
        if not match:
            continue
        prefix = candidate[: match.start()].rstrip()
        suffix = candidate[match.start():].strip()
        if not prefix or not suffix:
            continue
        if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", prefix):
            continue
        if answer_uses_unfriendly_math_format(suffix):
            return prefix
    return candidate


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
        self._cv_scope_cache: Dict[str, bool] = {}
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
        raw_text = clean_display_text(text)
        expanded = raw_text.lower()
        for pattern, canonical in COURSE_QUERY_ALIAS_RULES:
            def repl(match: re.Match) -> str:
                matched = clean_display_text(match.group(0)).lower().strip()
                if canonical in matched:
                    return matched
                return f"{matched} {canonical}".strip()

            expanded = pattern.sub(repl, expanded)

        def cv_repl(match: re.Match) -> str:
            matched = clean_display_text(match.group(0)).lower().strip()
            if "computer vision" in matched:
                return matched

            surrounding_tokens = {
                token
                for token in tokenize(expanded)
                if len(token) > 1
            }
            if surrounding_tokens & CV_RESUME_CONTEXT_TOKENS:
                return matched

            cv_domain_tokens = (CV_DOMAIN_TOKENS | COURSE_CONTEXT_HINTS | {"vision", "lecture", "lectures", "course"})
            if surrounding_tokens & cv_domain_tokens:
                return f"{matched} computer vision".strip()

            if re.fullmatch(r"\s*(?:what is|define|explain|introduce|compare)\s+cv\s*[?.!]*\s*", expanded):
                return f"{matched} computer vision".strip()

            if len(surrounding_tokens) <= 3:
                return f"{matched} computer vision".strip()

            return matched

        expanded = SAFE_CV_ALIAS_RE.sub(cv_repl, expanded)
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

    def _concept_tokens(self, text: str) -> List[str]:
        return [
            token
            for token in tokenize(self._expand_course_aliases(text))
            if len(token) > 1
            and token not in QUERY_NOISE_TOKENS
            and token not in QUERY_INTENT_TOKENS
            and token not in STRUCTURAL_QUERY_TOKENS
        ]

    def _query_target_concepts(
        self,
        query: str,
        kp_context: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        limit: int = 4,
    ) -> List[Dict[str, Any]]:
        lowered = self._expand_course_aliases(query)
        if not lowered:
            return []

        phrase_pool: set[str] = set(self.course_anchor_phrases) | set(CV_DOMAIN_PHRASES)
        for hit in list(kp_context or [])[:8]:
            item = hit["item"]
            name = clean_display_text(item.get("name", "")).lower()
            if len(name) >= 4:
                phrase_pool.add(name)
            for keyword in item.get("keywords", []):
                cleaned = clean_display_text(keyword).lower()
                if len(cleaned) >= 4:
                    phrase_pool.add(cleaned)

        phrase_hits: List[Tuple[int, int, str, List[str]]] = []
        for phrase in phrase_pool:
            cleaned_phrase = clean_display_text(phrase).lower()
            if len(cleaned_phrase) < 4:
                continue
            match = re.search(
                rf"(?<![A-Za-z0-9]){re.escape(cleaned_phrase)}(?![A-Za-z0-9])",
                lowered,
            )
            if not match:
                continue
            concept_tokens = self._concept_tokens(cleaned_phrase)
            if not concept_tokens:
                continue
            phrase_hits.append((match.start(), match.end(), cleaned_phrase, concept_tokens))

        phrase_hits.sort(key=lambda row: (row[0], -(row[1] - row[0]), row[2]))
        concepts: List[Dict[str, Any]] = []
        seen_labels: set[str] = set()
        occupied_spans: List[Tuple[int, int]] = []
        covered_tokens: set[str] = set()

        for start, end, phrase, concept_tokens in phrase_hits:
            if any(not (end <= left or start >= right) for left, right in occupied_spans):
                continue
            label = clean_display_text(phrase)
            key = label.lower()
            if not key or key in seen_labels:
                continue
            seen_labels.add(key)
            occupied_spans.append((start, end))
            covered_tokens.update(concept_tokens)
            concepts.append(
                {
                    "label": label,
                    "phrases": {phrase},
                    "tokens": set(concept_tokens),
                    "position": start,
                    "source": "phrase",
                }
            )
            if len(concepts) >= limit:
                return concepts[:limit]

        query_is_cv = self._looks_cv_request_by_terms(query)
        for token in self._query_focus_tokens(query):
            if len(token) < 4 or token in covered_tokens or token in STRUCTURAL_QUERY_TOKENS:
                continue
            if token in WEAK_COURSE_QUERY_TOKENS:
                continue
            appears_in_kp_context = any(
                token in self.kp_term_sets.get(hit["item"]["kp_id"], set())
                for hit in list(kp_context or [])[:8]
            )
            if not appears_in_kp_context and token not in self.course_anchor_tokens and token not in CV_DOMAIN_TOKENS and token not in STRONG_SINGLE_TERM_ANCHORS:
                if not query_is_cv or len(token) < 5:
                    continue
            label = clean_display_text(token)
            key = label.lower()
            if not key or key in seen_labels:
                continue
            seen_labels.add(key)
            covered_tokens.add(token)
            concepts.append(
                {
                    "label": label,
                    "phrases": {token},
                    "tokens": {token},
                    "position": lowered.find(token),
                    "source": "token",
                }
            )
            if len(concepts) >= limit:
                break

        concepts.sort(key=lambda row: (int(row.get("position", 10 ** 6)), str(row.get("label", "")).lower()))
        return concepts[:limit]

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
            if overlap_ratio >= 0.84 and min(len(candidate_tokens), len(previous_tokens)) >= 12:
                return True
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

    def _query_requests_practice_question(self, query: str) -> bool:
        cleaned = clean_display_text(query)
        if not cleaned:
            return False
        lowered = self._expand_course_aliases(cleaned)
        if QUESTION_GENERATION_RE.search(lowered):
            return True
        request_nouns = ("question", "questions", "problem", "problems", "exercise", "exercises", "quiz", "quizzes")
        request_verbs = ("give", "make", "create", "write", "design", "generate", "prepare", "set", "ask", "test")
        if any(noun in lowered for noun in request_nouns) and any(verb in lowered for verb in request_verbs):
            return True
        if "question" in lowered and any(token in lowered for token in ("practice", "quiz", "exercise", "problem", "test me")):
            return True
        return False

    def _practice_question_mode(self, query: str) -> str:
        lowered = self._expand_course_aliases(query)
        if MULTIPLE_CHOICE_REQUEST_RE.search(lowered):
            return "multiple_choice"
        if CALCULATION_QUESTION_RE.search(lowered):
            return "calculation"
        return "open"

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

    def _concept_matches_kp_item(self, concept: Dict[str, Any], item: Dict[str, Any]) -> bool:
        concept_phrases = {
            clean_display_text(phrase).lower()
            for phrase in concept.get("phrases", set())
            if clean_display_text(phrase)
        }
        concept_tokens = {
            token
            for token in concept.get("tokens", set())
            if clean_display_text(token)
        }
        reference_terms = self._kp_reference_terms(item)
        kp_phrases = {
            clean_display_text(phrase).lower()
            for phrase in reference_terms["phrases"]
            if clean_display_text(phrase)
        }
        if any(
            phrase in kp_phrase or kp_phrase in phrase
            for phrase in concept_phrases
            for kp_phrase in kp_phrases
        ):
            return True
        overlap = len(concept_tokens & reference_terms["tokens"])
        if len(concept_tokens) >= 2:
            return overlap >= max(2, len(concept_tokens) - 1)
        return overlap >= 1

    def _concept_matches_chunk_item(self, concept: Dict[str, Any], item: Dict[str, Any]) -> bool:
        chunk_id = str(item.get("chunk_id") or "")
        if not chunk_id:
            return False
        blob = self.chunk_search_blobs.get(chunk_id, "")
        title_text = clean_display_text(item.get("title", "")).lower()
        concept_phrases = {
            clean_display_text(phrase).lower()
            for phrase in concept.get("phrases", set())
            if clean_display_text(phrase)
        }
        if any(phrase in blob or phrase in title_text for phrase in concept_phrases):
            return True
        concept_tokens = {
            token
            for token in concept.get("tokens", set())
            if clean_display_text(token)
        }
        chunk_terms = self.chunk_term_sets.get(chunk_id, set()) | self.chunk_title_term_sets.get(chunk_id, set())
        overlap = len(concept_tokens & chunk_terms)
        if len(concept_tokens) >= 2:
            return overlap >= max(2, len(concept_tokens) - 1)
        return overlap >= 1

    def _concept_chunk_match_score(self, concept: Dict[str, Any], item: Dict[str, Any], base_score: float) -> float:
        chunk_id = str(item.get("chunk_id") or "")
        if not chunk_id:
            return float("-inf")
        blob = self.chunk_search_blobs.get(chunk_id, "")
        title_text = clean_display_text(item.get("title", "")).lower()
        concept_phrases = {
            clean_display_text(phrase).lower()
            for phrase in concept.get("phrases", set())
            if clean_display_text(phrase)
        }
        concept_tokens = {
            token
            for token in concept.get("tokens", set())
            if clean_display_text(token)
        }
        chunk_terms = self.chunk_term_sets.get(chunk_id, set()) | self.chunk_title_term_sets.get(chunk_id, set())
        phrase_hit = any(phrase in blob for phrase in concept_phrases)
        title_phrase_hit = any(phrase in title_text for phrase in concept_phrases)
        overlap = len(concept_tokens & chunk_terms)
        return (
            float(base_score)
            + (20.0 if phrase_hit else 0.0)
            + (6.0 if title_phrase_hit else 0.0)
            + overlap * 5.5
            + (2.0 if not self._is_low_priority_source(item) else 0.0)
        )

    def _covered_target_concepts_in_chunks(
        self,
        target_concepts: Sequence[Dict[str, Any]],
        chunk_hits: Sequence[Dict[str, Any]],
        *,
        limit: int = 4,
    ) -> List[str]:
        covered: List[str] = []
        seen_labels: set[str] = set()
        for concept in target_concepts:
            label = clean_display_text(concept.get("label", ""))
            if not label or label.lower() in seen_labels:
                continue
            for hit in list(chunk_hits or [])[:limit]:
                item = hit["item"]
                if self._is_low_priority_source(item):
                    continue
                if self._concept_matches_chunk_item(concept, item):
                    seen_labels.add(label.lower())
                    covered.append(label)
                    break
        return covered

    def _covered_target_concepts_in_kps(
        self,
        target_concepts: Sequence[Dict[str, Any]],
        kp_hits: Sequence[Dict[str, Any]],
        *,
        limit: int = 4,
    ) -> List[str]:
        covered: List[str] = []
        seen_labels: set[str] = set()
        for concept in target_concepts:
            label = clean_display_text(concept.get("label", ""))
            if not label or label.lower() in seen_labels:
                continue
            for hit in list(kp_hits or [])[:limit]:
                if self._concept_matches_kp_item(concept, hit["item"]):
                    seen_labels.add(label.lower())
                    covered.append(label)
                    break
        return covered

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
        target_concepts = list(task_profile.get("target_concepts") or [])
        selected: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        for concept in target_concepts[:limit]:
            concept_candidates = [
                hit
                for hit in explicit_hits + hits
                if hit["item"]["kp_id"] not in seen_ids and self._concept_matches_kp_item(concept, hit["item"])
            ]
            if not concept_candidates:
                continue
            concept_candidates.sort(
                key=lambda hit: (
                    self._kp_query_position(query, hit["item"]),
                    -len(self._kp_reference_terms(hit["item"])["tokens"] & set(concept.get("tokens", set()))),
                    -float(hit.get("score", 0.0)),
                )
            )
            best_hit = concept_candidates[0]
            seen_ids.add(best_hit["item"]["kp_id"])
            selected.append(best_hit)
            if len(selected) >= limit:
                return selected[:limit]

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
        for hit in kp_hits[:8]:
            profile = self._kp_query_match_profile(query, hit["item"])
            if profile["exact_name"] or profile["exact_keyword_hits"] or profile["strong_token_hits"]:
                explicit_hits.append(hit)
        target_concepts = self._query_target_concepts(query, kp_hits, limit=4)

        is_comparison = bool(COMPARISON_RE.search(lowered))
        is_relation = bool(RELATION_RE.search(lowered)) and not is_comparison
        is_why_how = bool(WHY_HOW_RE.match(cleaned))
        is_application = bool(APPLICATION_RE.search(lowered))
        is_definition = self._definition_request(query)
        is_question_generation_request = self._query_requests_practice_question(query)
        practice_question_mode = self._practice_question_mode(query) if is_question_generation_request else "none"
        is_code_request = self._query_requests_code(query)
        code_request_mode = self._code_request_mode(query) if is_code_request else "none"

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
            or len(target_concepts) >= 2
        )
        is_simple_definition = bool(
            is_definition
            and not is_code_request
            and not is_question_generation_request
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
                "target_concepts": target_concepts,
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
            "is_question_generation_request": is_question_generation_request,
            "practice_question_mode": practice_question_mode,
            "is_code_request": is_code_request,
            "code_request_mode": code_request_mode,
            "is_code_only_request": code_request_mode == "code_only",
            "is_code_plus_explanation_request": code_request_mode == "code_plus_explanation",
            "needs_multi_concept_coverage": needs_multi_concept_coverage,
            "target_concepts": target_concepts,
            "explicit_kp_hits": explicit_hits,
            "relevant_kp_hits": relevant_kp_hits,
            "answer_style": answer_style,
        }

    def _task_prompt_guidance(self, task_profile: Dict[str, Any]) -> str:
        target_labels = [
            clean_display_text(item.get("label", ""))
            for item in task_profile.get("target_concepts", [])
            if clean_display_text(item.get("label", ""))
        ]
        target_line = f" Target concepts: {', '.join(target_labels[:4])}." if target_labels else ""
        if task_profile.get("is_question_generation_request"):
            if task_profile.get("practice_question_mode") == "multiple_choice":
                return (
                    "The student wants a fresh practice question, not an explanation. Produce one lecturer-style multiple-choice question only, with four options and no solution."
                    + target_line
                )
            if task_profile.get("practice_question_mode") == "calculation":
                return (
                    "The student wants a fresh calculation practice question, not an explanation. Produce one small numeric exercise only, with explicit numbers and no worked solution."
                    + target_line
                )
            return (
                "The student wants a fresh practice question, not an explanation. Produce one realistic lecturer-style practice question only, without giving the solution."
                + target_line
            )
        if task_profile.get("is_code_only_request"):
            return (
                "This is a code-only request. Return only the code as a fenced Markdown code block in the requested language when possible. "
                "Do not add concept explanation, notes, bullets, or a source sentence outside the code block."
                + target_line
            )
        if task_profile.get("is_code_plus_explanation_request"):
            return (
                "This request asks for both explanation and code. Give a short direct explanation first, then a fenced Markdown code block. "
                "Keep the explanation brief and make the code the main deliverable."
                + target_line
            )
        if task_profile.get("is_comparison"):
            return (
                "This is a comparison question. Explicitly explain both concepts and state at least one concrete difference "
                "in purpose, input/output, or role in the vision pipeline. Do not answer only one side."
                + target_line
            )
        if task_profile.get("is_relation"):
            return (
                "This is a relationship question. Explain how the named concepts connect, and make sure each concept is addressed explicitly."
                + target_line
            )
        if task_profile.get("is_why_how"):
            return "This is a how/why question. Explain the mechanism or reason, not just the definition." + target_line
        if task_profile.get("is_application"):
            return "This is an application question. Explain what the concept is used for and why." + target_line
        if task_profile.get("is_simple_definition"):
            return "This is a single-concept definition question. A concise revision-style answer is appropriate." + target_line
        return "Answer the student's question directly and make sure all named concepts are covered." + target_line

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
        if task_profile.get("is_code_request"):
            has_code_block = self._answer_contains_code_block(body)
            if not has_code_block:
                return False
            prose_outside_code = self._strip_code_blocks_for_validation(body)
            prose_word_count = self._meaningful_prose_word_count(prose_outside_code)
            if task_profile.get("is_code_only_request"):
                if prose_word_count > 0:
                    return False
            elif task_profile.get("is_code_plus_explanation_request"):
                if prose_word_count < 6:
                    return False

        target_concepts = list(task_profile.get("target_concepts") or [])
        if task_profile.get("needs_multi_concept_coverage") and len(target_concepts) >= 2:
            answer_tokens = set(self._content_tokens(body))
            covered = 0
            for concept in target_concepts[:2]:
                concept_phrases = {
                    clean_display_text(phrase).lower()
                    for phrase in concept.get("phrases", set())
                    if clean_display_text(phrase)
                }
                concept_tokens = {
                    token
                    for token in concept.get("tokens", set())
                    if clean_display_text(token)
                }
                if any(phrase in self._expand_course_aliases(body) for phrase in concept_phrases):
                    covered += 1
                    continue
                overlap = len(answer_tokens & concept_tokens)
                if len(concept_tokens) >= 2 and overlap >= max(2, len(concept_tokens) - 1):
                    covered += 1
                elif len(concept_tokens) == 1 and overlap >= 1:
                    covered += 1
            if covered < 2:
                return False
        else:
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

    def _looks_cv_request_by_terms(self, query: str) -> bool:
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

    def _parse_cv_scope_payload(self, content: str) -> Optional[bool]:
        candidate = clean_display_text(content).strip()
        if not candidate:
            return None
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"\s*```$", "", candidate).strip()
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict) and "is_cv_related" in payload:
                return bool(payload.get("is_cv_related"))
        except Exception:
            pass
        match = re.search(r'"is_cv_related"\s*:\s*(true|false)', candidate, re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true"
        lowered = candidate.lower()
        if lowered in {"true", "yes", "cv", "computer vision"}:
            return True
        if lowered in {"false", "no", "not cv", "not computer vision"}:
            return False
        return None

    def _llm_cv_scope_decision(self, query: str) -> Optional[bool]:
        cleaned = clean_display_text(query)
        if not cleaned or self._llm_client is None or not self._llm_model:
            return None
        cache_key = cleaned.lower()
        if cache_key in self._cv_scope_cache:
            return self._cv_scope_cache[cache_key]

        prompt = textwrap.dedent(
            f"""
            Decide whether the student's question is about computer vision or a closely related visual-computing topic.

            Return JSON only:
            {{"is_cv_related": true}}
            or
            {{"is_cv_related": false}}

            Mark true for questions about:
            - computer vision, image or video processing, image filtering
            - feature extraction, matching, camera geometry, 3D vision, stereo, SfM
            - object detection, segmentation, tracking, recognition
            - neural rendering, NeRF, radiance fields, CNNs or other vision models
            - code requests when the code is clearly about a vision concept

            Mark false for:
            - politics, biography, weather, finance, or unrelated casual chat
            - generic programming questions with no visual-computing context

            If the question could reasonably be answered as part of a computer vision discussion, prefer true.

            Student question:
            {cleaned}
            """
        ).strip()

        try:
            response = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict JSON classifier. Reply with JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=40,
            )
            decision = self._parse_cv_scope_payload(response.choices[0].message.content or "")
            if decision is not None:
                self._cv_scope_cache[cache_key] = decision
                return decision
        except Exception:
            pass
        return None

    def _looks_cv_request(self, query: str) -> bool:
        llm_decision = self._llm_cv_scope_decision(query)
        if llm_decision is not None:
            return llm_decision
        return self._looks_cv_request_by_terms(query)

    def _course_coverage_level(
        self,
        query: str,
        chunk_hits: Sequence[Dict[str, Any]],
        *,
        task_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        focus_tokens = self._query_focus_tokens(query)
        focus_token_set = set(focus_tokens)
        focus_phrases = self._query_focus_phrases(query)
        target_concepts = list((task_profile or {}).get("target_concepts") or self._query_target_concepts(query))
        if target_concepts:
            covered = self._covered_target_concepts_in_chunks(target_concepts, chunk_hits, limit=4)
            if len(target_concepts) >= 2:
                if len(covered) >= min(2, len(target_concepts)):
                    return "direct"
                if covered:
                    return "related"
            elif covered:
                return "direct"
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
        *,
        task_profile: Optional[Dict[str, Any]] = None,
    ) -> bool:
        target_concepts = list((task_profile or {}).get("target_concepts") or self._query_target_concepts(query, kp_hits))
        if target_concepts:
            preferred_hits = self._preferred_chunk_hits(query, chunk_hits, limit=3, kp_context=kp_hits)
            covered_labels = set(self._covered_target_concepts_in_chunks(target_concepts, preferred_hits or chunk_hits, limit=4))
            covered_labels.update(self._covered_target_concepts_in_kps(target_concepts, kp_hits, limit=4))
            if len(target_concepts) >= 2:
                return len(covered_labels) >= min(2, len(target_concepts))
            if covered_labels:
                return True

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
        target_concepts = self._query_target_concepts(query, kp_context, limit=4)
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
            concept_matches = 0
            for concept in target_concepts:
                if self._concept_matches_chunk_item(concept, item):
                    concept_matches += 1
                    score += 7.2
            if len(target_concepts) >= 2 and concept_matches == 0:
                score -= 4.8
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
        *,
        task_profile: Optional[Dict[str, Any]] = None,
    ) -> bool:
        task_profile = task_profile or self._query_task_profile(query, kp_hits)
        anchor_details = self._query_anchor_details(query)
        preferred_hits = self._preferred_chunk_hits(query, chunk_hits, limit=3, kp_context=kp_hits)
        target_concepts = list(task_profile.get("target_concepts") or [])
        if target_concepts:
            covered_labels = set(self._covered_target_concepts_in_chunks(target_concepts, preferred_hits or chunk_hits, limit=4))
            covered_labels.update(self._covered_target_concepts_in_kps(target_concepts, kp_hits, limit=4))
            if len(target_concepts) >= 2 and len(covered_labels) < min(2, len(target_concepts)):
                return False
            if len(target_concepts) == 1 and not covered_labels:
                return False
        if not anchor_details["tokens"] and not anchor_details["phrases"]:
            return self._retrieval_supports_course_answer(query, kp_hits, preferred_hits or chunk_hits, task_profile=task_profile)

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
        return self._retrieval_supports_course_answer(query, kp_hits, preferred_hits or chunk_hits, task_profile=task_profile)

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
        task_profile: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not chunk_hits:
            return []
        task_profile = task_profile or self._query_task_profile(query)
        target_concepts = list(task_profile.get("target_concepts") or self._query_target_concepts(query, limit=4))
        if len(target_concepts) < 2:
            return self._preferred_chunk_hits(query, chunk_hits, limit=limit, kp_context=None)

        selected: List[Dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        for concept in target_concepts[:4]:
            candidates = [
                hit
                for hit in chunk_hits
                if str(hit["item"].get("chunk_id") or "")
                and str(hit["item"].get("chunk_id") or "") not in seen_chunk_ids
                and self._concept_matches_chunk_item(concept, hit["item"])
            ]
            if not candidates:
                continue
            candidates.sort(
                key=lambda hit: (
                    -self._concept_chunk_match_score(concept, hit["item"], float(hit.get("score", 0.0))),
                    self._is_low_priority_source(hit["item"]),
                )
            )
            best_hit = candidates[0]
            selected.append(best_hit)
            seen_chunk_ids.add(str(best_hit["item"].get("chunk_id") or ""))

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

    def _query_has_strong_focus(self, query: str) -> bool:
        lowered = self._expand_course_aliases(query)
        anchor_details = self._query_anchor_details(query)
        focus_tokens = self._query_focus_tokens(query)
        if any(topic in lowered for topic in CORE_COURSE_TOPICS):
            return True
        if any(phrase in lowered for phrase in CV_DOMAIN_PHRASES):
            return True
        if anchor_details["phrases"]:
            return True
        if anchor_details["tokens"] & STRONG_SINGLE_TERM_ANCHORS:
            return True
        if focus_tokens and self._looks_cv_request_by_terms(query):
            return True
        return False

    def _query_language_hint(self, text: str) -> str:
        lowered = clean_display_text(text).lower()
        if "c++" in lowered or re.search(r"\bcpp\b", lowered):
            return "C++"
        if "python" in lowered:
            return "Python"
        if "pytorch" in lowered:
            return "PyTorch"
        if "java" in lowered:
            return "Java"
        if "javascript" in lowered or re.search(r"\bjs\b", lowered):
            return "JavaScript"
        return ""

    def _paired_focus_from_query(self, query: str) -> List[str]:
        cleaned = clean_display_text(query)
        if not cleaned:
            return []
        patterns = [
            re.compile(
                r"\b(?:difference|differences|compare|comparison|contrast|relation|relationship|connection|link)\b"
                r".*?\bbetween\s+(?P<left>[A-Za-z0-9 +._/-]+?)\s+and\s+(?P<right>[A-Za-z0-9 +._/-]+?)(?:[?.!,]|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bcompare\s+(?P<left>[A-Za-z0-9 +._/-]+?)\s+and\s+(?P<right>[A-Za-z0-9 +._/-]+?)(?:[?.!,]|$)",
                re.IGNORECASE,
            ),
        ]
        for pattern in patterns:
            match = pattern.search(cleaned)
            if not match:
                continue
            left = clean_display_text(match.group("left")).strip(" .!?")
            right = clean_display_text(match.group("right")).strip(" .!?")
            if left and right:
                return [left, right]
        return []

    def _session_memory_state(self, session_memory: Optional[Dict[str, Any]]) -> Dict[str, str]:
        payload = dict(session_memory or {})
        return {
            "session_summary": clean_display_text(payload.get("session_summary", "")),
            "active_topic": clean_display_text(payload.get("active_topic", "")),
        }

    def _topic_signature(self, text: str) -> set[str]:
        lowered = self._expand_course_aliases(clean_display_text(text))
        if not lowered:
            return set()
        tokens = {
            token
            for token in tokenize(lowered)
            if len(token) > 2
            and token not in STOPWORDS
            and token not in QUERY_INTENT_TOKENS
            and token not in FOLLOWUP_NON_TOPIC_TOKENS
            and token
            not in {
                "course",
                "chat",
                "student",
                "students",
                "assistant",
                "discussion",
                "understanding",
                "learning",
                "latest",
                "recent",
                "question",
                "questions",
                "focus",
                "current",
            }
        }
        for phrase in self._query_focus_phrases(lowered):
            tokens.update(
                token
                for token in tokenize(phrase)
                if len(token) > 2 and token not in STOPWORDS
            )
        return tokens

    def _topic_shifted(self, previous_topic: str, new_topic: str) -> bool:
        previous_clean = clean_display_text(previous_topic).lower()
        new_clean = clean_display_text(new_topic).lower()
        if not previous_clean or not new_clean:
            return False
        if previous_clean == new_clean:
            return False
        previous_signature = self._topic_signature(previous_clean)
        new_signature = self._topic_signature(new_clean)
        if not previous_signature or not new_signature:
            return previous_clean != new_clean
        if previous_signature.issubset(new_signature) or new_signature.issubset(previous_signature):
            return False
        overlap = len(previous_signature & new_signature) / max(len(previous_signature), len(new_signature))
        return overlap < 0.5

    def _session_request_hint(self, raw_query: str, effective_query: str) -> str:
        expanded = self._expand_course_aliases(clean_display_text(effective_query or raw_query))
        if self._query_requests_code(raw_query):
            return "The student wants a code example or implementation next."
        if self._query_requests_practice_question(raw_query):
            return "The student wants a follow-up practice question."
        if COMPARISON_RE.search(expanded):
            return "The student is comparing closely related concepts."
        if RELATION_RE.search(expanded):
            return "The student is asking how concepts connect."
        if WHY_HOW_RE.search(clean_display_text(raw_query)):
            return "The student wants intuition or reasoning, not just a definition."
        return ""

    def _compose_session_memory_summary(
        self,
        raw_query: str,
        effective_query: str,
        answer: str,
        previous_memory: Optional[Dict[str, Any]],
        *,
        active_topic: str,
        llm_summary: str = "",
    ) -> str:
        previous = self._session_memory_state(previous_memory)
        request_hint = self._session_request_hint(raw_query, effective_query)
        answer_hint = safe_snippet(clean_markdown_text(extract_student_answer_text(answer)), 180)

        parts: List[str] = []
        if active_topic:
            parts.append(f"Current focus: {active_topic}.")
        if self._topic_shifted(previous["active_topic"], active_topic):
            parts.append(f"The discussion has moved on from {previous['active_topic']}.")
        if request_hint:
            parts.append(request_hint)
        if effective_query:
            parts.append(f"Latest standalone question: {safe_snippet(effective_query, 180)}.")
        if answer_hint and not self._query_requests_code(raw_query):
            parts.append(f"Latest answer gist: {answer_hint}.")
        if not parts and llm_summary:
            parts.append(clean_display_text(llm_summary))

        summary = clean_display_text(" ".join(part for part in parts if part))
        if not summary:
            summary = previous["session_summary"]
        return summary[:560]

    def _query_concept_labels(self, query: str, *, limit: int = 4) -> List[str]:
        return [
            clean_display_text(item.get("label", ""))
            for item in self._query_target_concepts(query, limit=limit)
            if clean_display_text(item.get("label", ""))
        ]

    def _derive_active_topic(self, query: str, fallback: str = "") -> str:
        cleaned = clean_display_text(query)
        if not cleaned:
            return clean_display_text(fallback)
        concept_labels = self._query_concept_labels(cleaned, limit=2)
        if len(concept_labels) >= 2:
            return " and ".join(concept_labels[:2])
        if concept_labels:
            return concept_labels[0]
        paired = self._paired_focus_from_query(cleaned)
        if len(paired) >= 2:
            return " and ".join(paired[:2])
        focus_phrases = sorted(self._query_focus_phrases(cleaned), key=len, reverse=True)
        if focus_phrases:
            return clean_display_text(focus_phrases[0])
        focus_tokens = self._query_focus_tokens(cleaned)
        if focus_tokens:
            return clean_display_text(" ".join(focus_tokens[:4]))
        return clean_display_text(fallback)

    def _candidate_preserves_query_concepts(self, raw_query: str, candidate_query: str) -> bool:
        raw_concepts = list(self._query_target_concepts(raw_query, limit=4))
        if not raw_concepts:
            return True
        candidate_lower = self._expand_course_aliases(candidate_query)
        candidate_tokens = set(tokenize(candidate_lower))
        for concept in raw_concepts:
            concept_phrases = {
                clean_display_text(phrase).lower()
                for phrase in concept.get("phrases", set())
                if clean_display_text(phrase)
            }
            if any(phrase in candidate_lower for phrase in concept_phrases):
                continue
            concept_tokens = {
                clean_display_text(token).lower()
                for token in concept.get("tokens", set())
                if clean_display_text(token)
            }
            if concept_tokens and concept_tokens.issubset(candidate_tokens):
                continue
            return False
        return True

    def _context_resolution_candidate_valid(
        self,
        raw_query: str,
        candidate_query: str,
        *,
        followup_detected: bool,
    ) -> bool:
        raw = clean_display_text(raw_query)
        candidate = clean_display_text(candidate_query)
        if not raw or not candidate:
            return False
        if self._query_requests_code(raw) and not self._query_requests_code(candidate):
            return False
        if self._query_requests_practice_question(raw) and not self._query_requests_practice_question(candidate):
            return False
        raw_language = self._query_language_hint(raw)
        if raw_language and self._query_requests_code(raw) and self._query_language_hint(candidate) != raw_language:
            return False
        if not self._candidate_preserves_query_concepts(raw, candidate):
            return False
        if self._query_has_strong_focus(raw) and not followup_detected:
            return clean_display_text(candidate) == raw
        return True

    def _session_context_focus(
        self,
        history: Sequence[Dict[str, str]],
        session_memory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        focus = self._conversation_focus(history)
        memory_state = self._session_memory_state(session_memory)
        focus_concept = clean_display_text(focus.get("focus_concept", "")) or memory_state["active_topic"]
        focus_concepts = [
            clean_display_text(item)
            for item in (focus.get("focus_concepts") or [])
            if clean_display_text(item)
        ]
        if not focus_concepts and memory_state["active_topic"]:
            focus_concepts = [memory_state["active_topic"]]
        merged = dict(focus)
        merged["focus_concept"] = focus_concept
        merged["focus_concepts"] = focus_concepts
        merged["session_summary"] = memory_state["session_summary"]
        merged["active_topic"] = memory_state["active_topic"]
        return merged

    def _conversation_focus(self, history: Sequence[Dict[str, str]]) -> Dict[str, Any]:
        turns = list(history or [])
        if not turns:
            return {
                "anchor_query": "",
                "anchor_answer": "",
                "focus_concept": "",
                "focus_concepts": [],
                "language_hint": "",
                "used_turn_index": -1,
            }

        for index in range(len(turns) - 1, -1, -1):
            turn = turns[index]
            if str(turn.get("role") or "").lower() != "user":
                continue
            anchor_query = clean_display_text(turn.get("content", ""))
            if not anchor_query or self._assistant_intent(anchor_query):
                continue
            if not self._query_has_strong_focus(anchor_query):
                continue

            anchor_answer = ""
            for follow_turn in turns[index + 1 :]:
                follow_role = str(follow_turn.get("role") or "").lower()
                if follow_role == "assistant":
                    anchor_answer = clean_markdown_text(
                        extract_student_answer_text(str(follow_turn.get("content") or ""))
                    )
                    break
                if follow_role == "user":
                    break

            kp_hits = self.search_knowledge_points(anchor_query, top_k=8)
            task_profile = self._query_task_profile(anchor_query, kp_hits) if kp_hits else {
                "needs_multi_concept_coverage": False,
                "relevant_kp_hits": [],
                "target_concepts": [],
            }
            relevant_hits = list(task_profile.get("relevant_kp_hits") or []) or list(kp_hits[:2])
            target_concepts = [
                clean_display_text(item.get("label", ""))
                for item in (task_profile.get("target_concepts") or [])
                if clean_display_text(item.get("label", ""))
            ]
            focus_concepts = list(target_concepts[:2])
            paired_focus = self._paired_focus_from_query(anchor_query)
            if task_profile.get("needs_multi_concept_coverage") and len(focus_concepts) < 2 and len(paired_focus) >= 2:
                focus_concepts = paired_focus
            if not focus_concepts:
                focus_concepts = [
                    clean_display_text(hit["item"].get("name", ""))
                    for hit in relevant_hits[:2]
                    if clean_display_text(hit["item"].get("name", ""))
                ]
            if not focus_concepts:
                phrases = sorted(self._query_focus_phrases(anchor_query), key=len, reverse=True)
                if phrases:
                    focus_concepts = [phrases[0]]
                else:
                    focus_tokens = self._query_focus_tokens(anchor_query)
                    if focus_tokens:
                        focus_concepts = [" ".join(focus_tokens[:4])]

            focus_concept = ""
            if task_profile.get("needs_multi_concept_coverage") and len(focus_concepts) >= 2:
                focus_concept = " and ".join(focus_concepts[:2])
            elif focus_concepts:
                focus_concept = focus_concepts[0]

            return {
                "anchor_query": anchor_query,
                "anchor_answer": safe_snippet(anchor_answer, 260),
                "focus_concept": focus_concept,
                "focus_concepts": focus_concepts,
                "language_hint": self._query_language_hint(anchor_query) or self._query_language_hint(anchor_answer),
                "used_turn_index": index,
            }

        return {
            "anchor_query": "",
            "anchor_answer": "",
            "focus_concept": "",
            "focus_concepts": [],
            "language_hint": "",
            "used_turn_index": -1,
        }

    def _semantic_followup_tokens(self, query: str) -> List[str]:
        return [
            token
            for token in self._query_focus_tokens(query)
            if token.lower() not in FOLLOWUP_NON_TOPIC_TOKENS
            and token.lower() not in STRUCTURAL_QUERY_TOKENS
        ]

    def _is_context_dependent_short_query(self, query: str) -> bool:
        current = clean_display_text(query)
        if not current:
            return False
        if self._query_has_strong_focus(current):
            return False

        lowered = self._expand_course_aliases(current)
        target_concepts = self._query_target_concepts(current, limit=3)
        semantic_focus_tokens = self._semantic_followup_tokens(current)
        short_query = len(self._content_tokens(current)) <= 10
        has_followup_shape = bool(
            FOLLOWUP_ACTION_RE.search(lowered)
            or FOLLOWUP_REFERENCE_RE.search(lowered)
            or REFERENCE_FOLLOWUP_RE.search(lowered)
            or self._query_requests_code(current)
            or self._query_requests_practice_question(current)
            or APPLICATION_RE.search(lowered)
            or COMPARISON_RE.search(lowered)
            or RELATION_RE.search(lowered)
            or WHY_HOW_RE.match(current)
        )
        if not short_query or not has_followup_shape:
            return False
        return not target_concepts and len(semantic_focus_tokens) == 0

    def _detect_followup_query(self, query: str, history: Sequence[Dict[str, str]]) -> bool:
        current = clean_display_text(query)
        if not current or not history:
            return False
        if self._assistant_intent(current):
            return False
        if self._query_has_strong_focus(current):
            return False

        lowered = self._expand_course_aliases(current)
        semantic_focus_tokens = self._semantic_followup_tokens(current)
        content_tokens = self._content_tokens(current)
        starts_like_followup = bool(FOLLOWUP_ACTION_RE.search(lowered))
        has_reference_words = bool(FOLLOWUP_REFERENCE_RE.search(lowered))
        has_code_or_example_shape = bool(
            self._query_requests_code(current)
            or self._query_requests_practice_question(current)
            or APPLICATION_RE.search(lowered)
            or COMPARISON_RE.search(lowered)
            or RELATION_RE.search(lowered)
            or WHY_HOW_RE.match(current)
        )
        short_query = len(content_tokens) <= 8
        if has_reference_words and short_query:
            return True
        if starts_like_followup and short_query:
            return True
        if has_code_or_example_shape and len(semantic_focus_tokens) == 0:
            return True
        if lowered.startswith(("and ", "also ", "then ", "so ", "what about ", "how about ")):
            return True
        if self._is_context_dependent_short_query(current):
            return True
        return False

    def _rule_based_followup_resolution(self, query: str, focus: Dict[str, Any]) -> str:
        current = clean_display_text(query)
        if not current:
            return ""
        focus_concept = clean_display_text(focus.get("focus_concept", ""))
        if not focus_concept:
            return current

        lowered = self._expand_course_aliases(current)
        language_hint = self._query_language_hint(current) or clean_display_text(focus.get("language_hint", ""))
        if self._query_requests_practice_question(current):
            practice_mode = self._practice_question_mode(current)
            if practice_mode == "multiple_choice":
                return f"Give me one multiple-choice practice question about {focus_concept}."
            if practice_mode == "calculation":
                return f"Give me one calculation practice question about {focus_concept}."
            return f"Give me one practice question about {focus_concept}."
        if self._query_requests_code(current):
            if "explain" in lowered:
                if language_hint:
                    return f"Give me a {language_hint} code example to explain {focus_concept}."
                return f"Give me a code example to explain {focus_concept}."
            if language_hint:
                return f"Give me a {language_hint} code example for {focus_concept}."
            return f"Give me a code example for {focus_concept}."
        if WHY_HOW_RE.match(current):
            keyword = WHY_HOW_RE.match(current).group(1).lower()
            if keyword == "why":
                return f"Why is {focus_concept} important?"
            return f"How does {focus_concept} work?"
        if COMPARISON_RE.search(lowered) or RELATION_RE.search(lowered):
            return f"Regarding {focus_concept}, {current}"
        if APPLICATION_RE.search(lowered):
            if " and " in focus_concept:
                return f"Show me an example comparing {focus_concept}."
            return f"Show me an example of {focus_concept}."
        if FOLLOWUP_REFERENCE_RE.search(lowered):
            return f"Regarding {focus_concept}, {current}"
        return f"Regarding {focus_concept}, {current}"

    def _followup_needs_topic_injection(self, query: str, focus_concept: str) -> bool:
        current = clean_display_text(query)
        if not current or not clean_display_text(focus_concept):
            return False
        if self._query_has_strong_focus(current):
            return False
        if self._query_target_concepts(current, limit=3):
            return False

        if self._is_context_dependent_short_query(current):
            return True

        lowered = self._expand_course_aliases(current)
        has_followup_shape = bool(
            FOLLOWUP_ACTION_RE.search(lowered)
            or FOLLOWUP_REFERENCE_RE.search(lowered)
            or REFERENCE_FOLLOWUP_RE.search(lowered)
            or self._query_requests_code(current)
            or self._query_requests_practice_question(current)
            or APPLICATION_RE.search(lowered)
            or COMPARISON_RE.search(lowered)
            or RELATION_RE.search(lowered)
            or WHY_HOW_RE.match(current)
        )
        if not has_followup_shape:
            return False

        semantic_focus_tokens = [
            token
            for token in self._semantic_followup_tokens(current)
        ]
        return len(semantic_focus_tokens) == 0

    def _llm_resolve_query_with_context(
        self,
        query: str,
        history: Sequence[Dict[str, str]],
        focus: Dict[str, Any],
        session_memory: Optional[Dict[str, Any]],
        rule_based_candidate: str,
    ) -> Optional[Dict[str, Any]]:
        cleaned = clean_display_text(query)
        if not cleaned or self._llm_client is None or not self._llm_model:
            return None

        memory_state = self._session_memory_state(session_memory)
        recent_history = "\n".join(
            f"{turn.get('role', 'user')}: {clean_display_text(turn.get('content', ''))}"
            for turn in list(history)[-6:]
            if clean_display_text(turn.get("content", ""))
        )
        prompt = textwrap.dedent(
            f"""
            Rewrite the student's latest message into a standalone question only if conversational history is genuinely needed.

            Return strict JSON only:
            {{"resolved_query": "...", "focus_concept": "...", "used_history": true}}

            Rules:
            - Do not answer the question.
            - Preserve the student's requested output style, language, and programming-language hint.
            - If the latest message is already a clear standalone question, keep it unchanged and set used_history to false.
            - If the latest message is unrelated to the previous course topic, keep it unchanged and set used_history to false.
            - Use the recent conversation and session summary to reconstruct missing topic information when the latest message depends on earlier turns.
            - If the latest message is a short follow-up such as "show me code", "why", "how", "give me an example", or "give me a question", rewrite it into a standalone question using the conversation focus.
            - If the latest message already names a new topic clearly, do not carry the old topic over.
            - Keep the rewritten question natural and concise.

            Latest student message:
            {cleaned}

            Session memory:
            - session_summary: {memory_state['session_summary'] or 'None'}
            - active_topic: {memory_state['active_topic'] or 'None'}

            Conversation focus:
            - anchor_query: {clean_display_text(focus.get('anchor_query', '')) or 'None'}
            - focus_concept: {clean_display_text(focus.get('focus_concept', '')) or 'None'}
            - anchor_answer_summary: {clean_display_text(focus.get('anchor_answer', '')) or 'None'}

            Rule-based candidate:
            {clean_display_text(rule_based_candidate) or cleaned}

            Recent conversation:
            {recent_history or 'None'}
            """
        ).strip()

        try:
            response = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You rewrite ambiguous follow-up questions into standalone course questions. Reply with JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=140,
            )
            parsed = parse_json_response(response.choices[0].message.content or "")
            if not isinstance(parsed, dict):
                return None
            resolved_query = clean_display_text(parsed.get("resolved_query", ""))
            focus_concept = clean_display_text(parsed.get("focus_concept", ""))
            used_history = bool(parsed.get("used_history") or parsed.get("used_context"))
            if not resolved_query:
                return None
            return {
                "resolved_query": resolved_query,
                "focus_concept": focus_concept or clean_display_text(focus.get("focus_concept", "")),
                "used_history": used_history,
            }
        except Exception:
            return None

    def _resolve_query_with_context(
        self,
        query: str,
        history: Sequence[Dict[str, str]],
        session_memory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        current = clean_display_text(query)
        memory_state = self._session_memory_state(session_memory)
        base = {
            "raw_query": current,
            "resolved_query": current,
            "followup_detected": False,
            "used_history": False,
            "focus_concept": "",
            "anchor_query": "",
            "anchor_answer": "",
            "language_hint": self._query_language_hint(current),
            "session_summary": memory_state["session_summary"],
            "active_topic": memory_state["active_topic"],
        }
        if not current:
            return base
        if self._assistant_intent(current):
            return base

        followup_detected = self._detect_followup_query(current, history)
        focus = self._session_context_focus(history, session_memory)
        focus_concept = clean_display_text(focus.get("focus_concept", ""))
        rule_based_candidate = self._rule_based_followup_resolution(current, focus)
        underspecified = self._followup_needs_topic_injection(current, focus_concept)
        strong_self_contained_query = self._query_has_strong_focus(current) and not followup_detected

        resolved_query = current
        used_history = False
        has_session_context = bool(history or memory_state["session_summary"] or focus_concept)
        if has_session_context and not strong_self_contained_query:
            llm_resolution = self._llm_resolve_query_with_context(
                current,
                history,
                focus,
                session_memory,
                rule_based_candidate,
            )
            if llm_resolution:
                candidate = clean_display_text(llm_resolution.get("resolved_query", ""))
                if candidate and self._context_resolution_candidate_valid(
                    current,
                    candidate,
                    followup_detected=followup_detected,
                ):
                    resolved_query = candidate
                    used_history = bool(llm_resolution.get("used_history"))
                    focus_concept = clean_display_text(llm_resolution.get("focus_concept", "")) or focus_concept
            if (
                resolved_query == current
                and underspecified
                and clean_display_text(rule_based_candidate)
                and clean_display_text(rule_based_candidate) != current
            ):
                resolved_query = clean_display_text(rule_based_candidate)
                used_history = True

        active_topic = self._derive_active_topic(
            resolved_query,
            fallback=focus_concept or memory_state["active_topic"],
        )

        return {
            "raw_query": current,
            "resolved_query": clean_display_text(resolved_query or current),
            "followup_detected": followup_detected,
            "used_history": bool(used_history and clean_display_text(resolved_query) != current),
            "focus_concept": active_topic or focus_concept,
            "anchor_query": clean_display_text(focus.get("anchor_query", "")),
            "anchor_answer": clean_display_text(focus.get("anchor_answer", "")),
            "language_hint": self._query_language_hint(current) or clean_display_text(focus.get("language_hint", "")),
            "session_summary": memory_state["session_summary"],
            "active_topic": active_topic,
        }

    def _contextual_query(
        self,
        query: str,
        history: Sequence[Dict[str, str]],
        session_memory: Optional[Dict[str, Any]] = None,
    ) -> str:
        resolved = self._resolve_query_with_context(query, history, session_memory=session_memory)
        return clean_display_text(resolved.get("resolved_query", "")) or clean_display_text(query)

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

    def _code_request_mode(self, query: str) -> str:
        lowered = self._expand_course_aliases(query)
        if not self._query_requests_code(query):
            return "none"
        explanation_markers = (
            "explain",
            "what is",
            "what are",
            "difference",
            "compare",
            "comparison",
            "intuition",
            "concept",
            "theory",
            "overview",
            "with explanation",
            "and explain",
            "plus explain",
            "walk me through",
            "tell me about",
        )
        if (
            DEFINITION_RE.search(lowered)
            or WHY_HOW_RE.match(clean_display_text(query))
            or COMPARISON_RE.search(lowered)
            or RELATION_RE.search(lowered)
            or APPLICATION_RE.search(lowered)
            or any(marker in lowered for marker in explanation_markers)
        ):
            return "code_plus_explanation"
        return "code_only"

    def _extract_fenced_code_blocks(self, text: str) -> List[str]:
        candidate = str(text or "")
        return [match.group(0).strip() for match in re.finditer(r"```[\s\S]*?```", candidate)]

    def _answer_contains_code_block(self, text: str) -> bool:
        return bool(self._extract_fenced_code_blocks(text))

    def _strip_code_blocks_for_validation(self, text: str) -> str:
        candidate = re.sub(r"```[\s\S]*?```", " ", str(text or ""))
        candidate = re.sub(r"(?im)^Course source:.*$", " ", candidate)
        return clean_markdown_text(candidate)

    def _code_block_quality_ok(self, text: str, query: str) -> bool:
        blocks = self._extract_fenced_code_blocks(text)
        if not blocks:
            return False
        primary = blocks[0]
        code = re.sub(r"^```[A-Za-z0-9_-]*\s*\n?", "", primary)
        code = re.sub(r"\n?```$", "", code).strip()
        lines = [line for line in code.splitlines() if line.strip()]
        if len(lines) < 3:
            return False
        if code.count("(") > code.count(")"):
            return False
        if code.count("[") > code.count("]"):
            return False
        if code.count("{") > code.count("}"):
            return False
        last_line = lines[-1].strip()
        if re.search(r"[\(\[\{=:,+\-*/]$", last_line):
            return False
        if re.search(r"\b(print|return|if|elif|for|while|with|def|class)\s*\(?$", last_line):
            return False
        lowered_query = clean_display_text(query).lower()
        lowered_code = code.lower()
        if ("python" in lowered_query or "pytorch" in lowered_query) and len(lines) < 5:
            strong_markers = ("import ", "def ", "class ", "torch", "nn.", "return ", "print(", "=")
            if not any(marker in lowered_code for marker in strong_markers):
                return False
        return True

    def _code_request_topic_issues(self, text: str, query: str) -> List[str]:
        blocks = self._extract_fenced_code_blocks(text)
        if not blocks:
            return []
        primary = blocks[0]
        code = re.sub(r"^```[A-Za-z0-9_-]*\s*\n?", "", primary)
        code = re.sub(r"\n?```$", "", code).strip()
        lowered_code = code.lower()
        lowered_query = self._expand_course_aliases(query)
        issues: List[str] = []

        if "..." in code:
            issues.append("The code example still contains placeholder ellipses instead of a concrete runnable example.")
        if re.search(r"\b[A-Za-z_][A-Za-z0-9_]*'(?=[,\]\)\s])", code):
            issues.append("The code example uses prime-style placeholder variable names that are not valid code.")

        if ("pytorch" in lowered_query or "torch" in lowered_query) and "convolution" in lowered_query:
            if "import torch" not in lowered_code:
                issues.append("A PyTorch example should import torch explicitly.")
            uses_2d_conv = "conv2d" in lowered_code or "nn.conv2d" in lowered_code
            uses_1d_conv = "conv1d" in lowered_code or "nn.conv1d" in lowered_code
            if uses_1d_conv and not uses_2d_conv:
                issues.append("For a computer-vision PyTorch convolution example, use a 2D image example instead of a 1D signal.")
            if "f.conv2d" in lowered_code or "torch.nn.functional.conv2d" in lowered_code:
                has_4d_input = (
                    "[[[[" in code
                    or ".view(1, 1," in lowered_code
                    or ".reshape(1, 1," in lowered_code
                    or ".unsqueeze(0).unsqueeze(0)" in lowered_code
                )
                if not has_4d_input:
                    issues.append("For torch.nn.functional.conv2d, the example input tensor should clearly use a valid 4D shape.")
            if "f.conv2d" in lowered_code and "[[[[" not in code and "kernel.view(1, 1," not in lowered_code and "weight.view(1, 1," not in lowered_code:
                issues.append("For torch.nn.functional.conv2d, the example kernel should clearly use a valid 4D weight shape.")
        return issues

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
        code_request_mode: str = "code_plus_explanation",
    ) -> str:
        if code_request_mode == "code_only":
            raw_body = clean_markdown_text(answer)
            if not raw_body:
                return raw_body
            code_blocks = self._extract_fenced_code_blocks(raw_body)
            if code_blocks:
                return "\n\n".join(block.strip() for block in code_blocks if block.strip()).strip()
            return raw_body.strip()
        body = promote_inline_code_blocks(answer, query=query)
        if not body:
            return body
        code_blocks = list(re.finditer(r"```[\s\S]*?```", body))
        if code_blocks:
            before = body[:code_blocks[0].start()].strip()
            after = body[code_blocks[-1].end():].strip()
            code_body = "\n\n".join(match.group(0).strip() for match in code_blocks if match.group(0).strip())
            has_intro = self._meaningful_prose_word_count(before) >= 6
            has_outro = self._meaningful_prose_word_count(after) >= 6
        else:
            code_body = ""
            has_intro = markdown_has_prose_outside_code(body)
            has_outro = True
        intro = self._code_answer_intro(query, related, chunk_hits)
        outro = self._code_answer_outro(query)
        parts: List[str] = []
        if code_blocks and has_intro and before:
            parts.append(before)
        elif not has_intro:
            parts.append(intro)
        parts.append(code_body or body.strip())
        if not has_outro:
            parts.append(outro)
        elif after:
            parts.append(after)
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

    def _citations_from_review_refs(
        self,
        review_refs: Sequence[Dict[str, Any]],
        *,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        seen_units: set[Tuple[str, str, int]] = set()
        for ref in self._dedupe_review_refs(review_refs)[: max(limit, 4)]:
            item = self._normalize_review_ref(ref)
            display_source = clean_display_text(item.get("display_source", ""))
            unit_type = clean_display_text(item.get("unit_type", ""))
            try:
                unit_index = int(item.get("unit_index", 0) or 0)
            except Exception:
                unit_index = 0
            unit_key = (normalized_path_key(display_source), unit_type, unit_index)
            if unit_key in seen_units:
                continue
            seen_units.add(unit_key)
            citations.append(
                {
                    "citation_id": f"S{len(citations) + 1}",
                    "source": clean_display_text(item.get("source", "")),
                    "display_source": display_source,
                    "location": clean_display_text(item.get("location", "")),
                    "unit_type": unit_type,
                    "unit_index": unit_index,
                    "chunk_index": int(item.get("chunk_index", 0) or 0),
                    "section": clean_display_text(item.get("section", "")) or clean_display_text(item.get("location", "")),
                    "snippet": clean_display_text(item.get("excerpt", "") or item.get("text", "")),
                }
            )
            if len(citations) >= limit:
                break
        return citations

    def _select_practice_question_from_bank(
        self,
        query: str,
        task_profile: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        relevant_hits = list(task_profile.get("relevant_kp_hits") or [])
        explicit_hits = list(task_profile.get("explicit_kp_hits") or [])
        target_concepts = list(task_profile.get("target_concepts") or [])

        candidate_kp_hits: List[Dict[str, Any]] = []
        seen_kp_ids: set[str] = set()

        def add_candidate(item: Dict[str, Any], score: float) -> None:
            kp_id = str(item.get("kp_id") or "")
            if not kp_id or kp_id in seen_kp_ids or not self.questions_by_kp.get(kp_id):
                return
            seen_kp_ids.add(kp_id)
            candidate_kp_hits.append({"item": item, "score": float(score)})

        for hit in relevant_hits[:6]:
            add_candidate(hit["item"], float(hit.get("score", 0.0)))
        for hit in explicit_hits[:6]:
            add_candidate(hit["item"], float(hit.get("score", 0.0)) + 3.0)

        for concept in target_concepts:
            concept_position = int(concept.get("position", 0) or 0)
            for item in self.kps:
                if not self._concept_matches_kp_item(concept, item):
                    continue
                query_profile = self._kp_query_match_profile(query, item)
                score = 14.0
                if query_profile["exact_name"]:
                    score += 4.0
                if query_profile["exact_keyword_hits"]:
                    score += 3.0 + 0.6 * len(query_profile["exact_keyword_hits"])
                if query_profile["strong_token_hits"]:
                    score += 2.0 + 0.5 * len(query_profile["strong_token_hits"])
                score -= min(concept_position, 120) * 0.01
                add_candidate(item, score)

        if not candidate_kp_hits:
            return None

        target_terms = {
            token
            for token in self._content_tokens(query)
            if token not in STRUCTURAL_QUERY_TOKENS and token not in QUERY_INTENT_TOKENS
        }
        target_terms.update(
            token
            for concept in target_concepts
            for token in concept.get("tokens", set())
            if clean_display_text(token)
        )
        mode = clean_display_text(task_profile.get("practice_question_mode", "")).lower()
        target_phrases = {
            clean_display_text(phrase).lower()
            for concept in target_concepts
            for phrase in concept.get("phrases", set())
            if clean_display_text(phrase)
        }
        kp_score_by_id = {
            str(hit["item"].get("kp_id") or ""): float(hit.get("score", 0.0))
            for hit in candidate_kp_hits
            if str(hit["item"].get("kp_id") or "")
        }
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for kp_id, questions in self.questions_by_kp.items():
            kp_boost = kp_score_by_id.get(kp_id, float("-inf"))
            if not target_terms and kp_id not in kp_score_by_id:
                continue
            for question in questions:
                question_text = clean_display_text(question.get("question", ""))
                if not question_text:
                    continue
                blob = clean_display_text(
                    "\n".join(
                        [
                            question_text,
                            clean_display_text(question.get("answer", "")),
                            clean_display_text(question.get("explanation", "")),
                            " ".join(clean_display_text(option) for option in question.get("options", [])),
                        ]
                    )
                ).lower()
                blob_terms = set(tokenize(blob))
                phrase_overlap = sum(1 for phrase in target_phrases if phrase and phrase in blob)
                overlap = len(target_terms & blob_terms) if target_terms else 0
                if target_terms and not phrase_overlap and overlap == 0 and kp_boost == float("-inf"):
                    continue
                score = phrase_overlap * 6.0 + overlap * 2.4
                if kp_boost != float("-inf"):
                    score += kp_boost * 0.18
                has_options = bool(question.get("options"))
                if mode == "multiple_choice":
                    score += 4.0 if has_options else -2.0
                elif mode == "calculation":
                    if CALCULATION_QUESTION_RE.search(blob) or any(marker in blob for marker in ("kernel", "filter", "matrix", "vector", "multiply", "compute", "output value")):
                        score += 4.5
                    if has_options:
                        score -= 0.6
                else:
                    score += 1.2 if not has_options else 0.8
                scored.append((score, question))

        if not scored:
            return None
        scored.sort(key=lambda row: row[0], reverse=True)
        best_score, best_question = scored[0]
        if best_score <= 0:
            return None
        return dict(best_question)

    def _format_bank_practice_question(
        self,
        question: Dict[str, Any],
        task_profile: Dict[str, Any],
    ) -> str:
        mode = clean_display_text(task_profile.get("practice_question_mode", "")).lower()
        parts: List[str] = ["Try this practice question:", "", clean_display_text(question.get("question", ""))]
        if mode == "multiple_choice":
            parsed_options = parse_options(question.get("options", []))
            if parsed_options:
                parts.extend(
                    f"{item['label']}. {item['text']}".strip()
                    for item in parsed_options[:4]
                    if item.get("text")
                )
            else:
                options = [clean_display_text(option) for option in question.get("options", []) if clean_display_text(option)]
                if options:
                    parts.extend(options[:4])
        parts.extend(["", "If you want, I can also check your answer or show the solution afterward."])
        return "\n".join(part for part in parts if part is not None).strip()

    def _normalize_practice_question_text(self, text: str, mode: str) -> str:
        cleaned = sanitize_transport_answer_text(extract_student_answer_text(text))
        if not cleaned:
            return ""
        if mode != "multiple_choice":
            return cleaned

        normalized_lines: List[str] = []
        for raw_line in cleaned.splitlines():
            line = raw_line.rstrip()
            match = PRACTICE_OPTION_LINE_RE.match(line)
            if not match:
                normalized_lines.append(line)
                continue
            label = match.group(1).upper()
            option_text = clean_display_text(match.group(2))
            normalized_lines.append(f"{label}. {option_text}" if option_text else f"{label}.")
        return "\n".join(normalized_lines).strip()

    def _practice_question_output_ok(self, text: str, mode: str) -> bool:
        cleaned = clean_display_text(self._normalize_practice_question_text(text, mode))
        if not cleaned:
            return False
        lowered = cleaned.lower()
        forbidden_markers = (
            "correct answer",
            "the answer is",
            "\nanswer:",
            "\nsolution:",
            "\nexplanation:",
        )
        if any(marker in lowered for marker in forbidden_markers):
            return False
        if mode == "multiple_choice":
            option_lines = [line for line in cleaned.splitlines() if PRACTICE_OPTION_LINE_RE.match(line)]
            return len(option_lines) >= 4
        if mode == "calculation":
            return bool(
                CALCULATION_QUESTION_RE.search(lowered)
                or any(marker in lowered for marker in ("kernel", "filter", "matrix", "vector", "output value", "compute"))
            )
        return True

    def _llm_practice_question(
        self,
        raw_query: str,
        history: Sequence[Dict[str, str]],
        related: Sequence[Dict[str, Any]],
        chunk_hits: Sequence[Dict[str, Any]],
        citations: Sequence[Dict[str, Any]],
        *,
        task_profile: Dict[str, Any],
        resolved_query: str,
        resolution_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.llm_enabled:
            raise ValueError("LLM is not configured.")

        context = self._llm_answer_context(
            raw_query,
            resolved_query,
            history,
            related,
            chunk_hits,
            citations,
            coverage_level="direct" if citations else "none",
            task_profile=task_profile,
            resolution_context=resolution_context,
        )
        mode = clean_display_text(task_profile.get("practice_question_mode", "")).lower() or "open"
        source_ids = ", ".join(context["allowed_source_ids"]) or "None"
        system_prompt = textwrap.dedent(
            """
            You create one student-facing practice question for a computer vision course.
            Return strict JSON with exactly two keys:
            - answer
            - used_sources
            The answer field must contain the question shown to the student.
            Do not provide the solution, answer key, or explanation.
            Do not mention file names, page numbers, source ids, or source lists in the answer body.
            """
        ).strip()
        user_prompt = textwrap.dedent(
            f"""
            Student request:
            {context['raw_query']}

            Standalone resolved request:
            {context['resolved_query']}

            Request mode:
            {mode}

            Target concepts:
            {context['target_concepts_text']}

            Relevant course topics:
            {context['relevant_topics']}

            Allowed source ids:
            {source_ids}

            Evidence:
            {context['evidence_text']}

            Requirements:
            - Produce exactly one fresh practice question, not an explanation.
            - Keep the wording natural and lecturer-like.
            - If mode is multiple_choice, provide four options labelled on separate lines exactly as A., B., C., and D., and do not reveal the answer.
            - If mode is calculation, produce one small numeric exercise with explicit values and do not solve it.
            - If mode is open, produce one short realistic practice question without the solution.
            - The final line may briefly tell the student that you can show the solution if they want.
            - Use used_sources only when the question is genuinely grounded in the supplied evidence.

            Return strict JSON only.
            """
        ).strip()
        raw_content = self._llm_json_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.22,
        )
        payload = self._parse_llm_answer_payload(
            raw_content,
            query=resolved_query,
            task_profile=task_profile,
            allowed_source_ids=context["allowed_source_ids"],
        )
        payload["answer"] = self._normalize_practice_question_text(payload.get("answer", ""), mode)
        if not self._practice_question_output_ok(payload.get("answer", ""), mode):
            raise ValueError("Generated practice question did not satisfy the requested format.")
        return payload

    def _practice_question_response(
        self,
        raw_query: str,
        history: Sequence[Dict[str, str]],
        related: Sequence[Dict[str, Any]],
        chunk_hits: Sequence[Dict[str, Any]],
        citations: Sequence[Dict[str, Any]],
        *,
        task_profile: Dict[str, Any],
        resolved_query: str,
        resolution_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.llm_enabled:
            try:
                payload = self._llm_practice_question(
                    raw_query,
                    history,
                    related,
                    chunk_hits,
                    citations,
                    task_profile=task_profile,
                    resolved_query=resolved_query,
                    resolution_context=resolution_context,
                )
                used_sources = payload.get("used_sources", [])
                final_citations = (
                    self._select_final_citations(citations, used_sources, max_count=3)
                    if used_sources
                    else list(citations[:2])
                )
                return {
                    "answer": self._answer_with_source_line(payload.get("answer", ""), final_citations),
                    "citations": final_citations,
                    "mode": "practice_question",
                }
            except Exception as exc:
                print(f"[Course4186KnowledgeBase] Practice question generation fallback: {exc}")

        bank_question = self._select_practice_question_from_bank(resolved_query, task_profile)
        if bank_question:
            final_citations = self._citations_from_review_refs(bank_question.get("review_refs", []), limit=3)
            return {
                "answer": self._answer_with_source_line(
                    self._format_bank_practice_question(bank_question, task_profile),
                    final_citations,
                ),
                "citations": final_citations,
                "mode": "practice_question",
            }

        concept_label = ""
        for concept in task_profile.get("target_concepts", []):
            concept_label = clean_display_text(concept.get("label", ""))
            if concept_label:
                break
        if not concept_label and related:
            concept_label = clean_display_text(related[0].get("name", ""))
        concept_label = concept_label or "this topic"
        mode = clean_display_text(task_profile.get("practice_question_mode", "")).lower()
        if mode == "calculation":
            fallback_text = (
                f"Try this practice question:\n\n"
                f"Use a small numeric example to calculate one step of {concept_label}. "
                f"State all input values clearly and work out the output by hand.\n\n"
                f"If you want, I can also turn this into a fully worked solution."
            )
        elif mode == "multiple_choice":
            fallback_text = (
                f"Try this practice question:\n\n"
                f"Which statement best matches {concept_label} in computer vision?\n"
                f"A. It is mainly used as a loss function for supervising image classification.\n"
                f"B. It is a geometric or visual representation that should be interpreted in relation to the lecture material.\n"
                f"C. It is only a data-storage format for saving camera parameters.\n"
                f"D. It is a hardware component inside the image sensor.\n\n"
                f"If you want, I can also prepare a more specific lecturer-style multiple-choice question."
            )
        else:
            fallback_text = (
                f"Try this practice question:\n\n"
                f"Explain {concept_label} in your own words and describe one key step, property, or use case that matters in practice.\n\n"
                f"If you want, I can also refine this into a more exam-style question."
            )
        return {
            "answer": self._answer_with_source_line(fallback_text, list(citations[:2])),
            "citations": list(citations[:2]),
            "mode": "practice_question",
        }

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

    def _chat_response(
        self,
        *,
        answer_text: str,
        citations: Sequence[Dict[str, Any]],
        related_kps: Sequence[Dict[str, Any]],
        mode: str,
        raw_query: str,
        history: Sequence[Dict[str, str]],
        previous_memory: Optional[Dict[str, Any]],
        resolved_query: str,
        resolution_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session_memory = self.build_session_memory(
            raw_query,
            answer_text,
            history,
            previous_memory=previous_memory,
            mode=mode,
            resolved_query=resolved_query,
            resolution_context=resolution_context,
            related=related_kps,
        )
        return {
            "answer": answer_text,
            "citations": list(citations or []),
            "related_kps": list(related_kps or []),
            "mode": mode,
            "session_memory": session_memory,
            "_resolved_query": clean_display_text(resolved_query),
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
        special_resolution = self._resolve_query_with_context(query, history, session_memory=session_memory)
        special_response = self._assistant_response_for_query(query, history)
        if special_response:
            return self._chat_response(
                answer_text=special_response,
                citations=[],
                related_kps=[],
                mode="assistant",
                raw_query=query,
                history=history,
                previous_memory=session_memory,
                resolved_query=clean_display_text(query),
                resolution_context=special_resolution,
            )

        resolved_context = self._resolve_query_with_context(query, history, session_memory=session_memory)
        resolved_query = clean_display_text(resolved_context.get("resolved_query", "")) or clean_display_text(query)

        if not self._looks_course_request(resolved_query):
            return self._chat_response(
                answer_text=self._out_of_scope_response(query, history),
                citations=[],
                related_kps=[],
                mode="assistant",
                raw_query=query,
                history=history,
                previous_memory=session_memory,
                resolved_query=resolved_query,
                resolution_context=resolved_context,
            )

        retrieval_query = resolved_query
        kp_hits = self.search_knowledge_points(retrieval_query, top_k=max(8, top_k + 3))
        task_profile = self._query_task_profile(retrieval_query, kp_hits)
        chunk_search_k = max(top_k, 8) if task_profile.get("needs_multi_concept_coverage") else top_k
        chunk_hits = self.search_chunks(retrieval_query, top_k=chunk_search_k, kp_context=kp_hits)
        focused_hits = self._focused_kp_support_hits(retrieval_query, kp_hits, limit=6, task_profile=task_profile)
        if task_profile.get("needs_multi_concept_coverage"):
            multi_concept_hits = self._multi_concept_chunk_hits(
                retrieval_query,
                chunk_hits,
                limit=4,
                task_profile=task_profile,
            )
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
        coverage_level = self._course_coverage_level(retrieval_query, citation_hits, task_profile=task_profile)
        related_source_hits = list(task_profile.get("relevant_kp_hits") or [])
        if not related_source_hits:
            related_source_hits = list(kp_hits[:3])
        related = [
            {
                "kp_id": hit["item"]["kp_id"],
                "name": hit["item"]["name"],
                "description": hit["item"]["description"],
            }
            for hit in related_source_hits[:3]
        ]

        if task_profile.get("is_question_generation_request"):
            practice_response = self._practice_question_response(
                query,
                history,
                related,
                citation_hits[:4],
                citations,
                task_profile=task_profile,
                resolved_query=resolved_query,
                resolution_context=resolved_context,
            )
            return self._chat_response(
                answer_text=practice_response["answer"],
                citations=list(practice_response.get("citations", [])),
                related_kps=related,
                mode=practice_response.get("mode", "practice_question"),
                raw_query=query,
                history=history,
                previous_memory=session_memory,
                resolved_query=resolved_query,
                resolution_context=resolved_context,
            )

        if not citations:
            if self._looks_cv_request(resolved_query) and self._llm_client is not None and self._llm_model:
                try:
                    llm_payload = self._llm_answer(
                        query,
                        history,
                        related,
                        [],
                        [],
                        coverage_level="none",
                        task_profile=task_profile,
                        resolved_query=resolved_query,
                        resolution_context=resolved_context,
                    )
                    answer_body = self._strip_uncited_course_connection(llm_payload.get("answer", ""))
                    return self._chat_response(
                        answer_text=self._answer_with_source_line(answer_body, []),
                        citations=[],
                        related_kps=related,
                        mode="llm",
                        raw_query=query,
                        history=history,
                        previous_memory=session_memory,
                        resolved_query=resolved_query,
                        resolution_context=resolved_context,
                    )
                except Exception as exc:
                    print(f"[Course4186KnowledgeBase] CV general answer fallback: {exc}")
            return self._chat_response(
                answer_text=(
                    "I could not find a clear answer for that in the current Course 4186 materials. "
                    "Try naming the lecture topic, method, or formula more specifically."
                ),
                citations=[],
                related_kps=related,
                mode="assistant",
                raw_query=query,
                history=history,
                previous_memory=session_memory,
                resolved_query=resolved_query,
                resolution_context=resolved_context,
            )

        if not self._is_grounded_answer_ready(retrieval_query, kp_hits, citation_hits, task_profile=task_profile):
            if self._looks_cv_request(resolved_query) and self._llm_client is not None and self._llm_model:
                try:
                    llm_payload = self._llm_answer(
                        query,
                        history,
                        related,
                        citation_hits[:4],
                        citations,
                        coverage_level=coverage_level,
                        task_profile=task_profile,
                        resolved_query=resolved_query,
                        resolution_context=resolved_context,
                    )
                    used_sources = llm_payload.get("used_sources", [])
                    if task_profile.get("is_code_only_request"):
                        final_citations = []
                    elif used_sources:
                        final_citations = self._select_final_citations(citations, used_sources, max_count=3)
                    elif task_profile.get("needs_multi_concept_coverage"):
                        final_citations = citations[:3]
                    else:
                        final_citations = []
                    answer_body = llm_payload.get("answer", "")
                    if not final_citations:
                        answer_body = self._strip_uncited_course_connection(answer_body)
                    return self._chat_response(
                        answer_text=self._answer_with_source_line(answer_body, final_citations),
                        citations=final_citations,
                        related_kps=related,
                        mode="llm",
                        raw_query=query,
                        history=history,
                        previous_memory=session_memory,
                        resolved_query=resolved_query,
                        resolution_context=resolved_context,
                    )
                except Exception as exc:
                    print(f"[Course4186KnowledgeBase] Partial CV answer fallback: {exc}")
                    if task_profile.get("is_code_request"):
                        try:
                            llm_payload = self._llm_answer(
                                query,
                                history,
                                related,
                                [],
                                [],
                                coverage_level="none",
                                task_profile=task_profile,
                                resolved_query=resolved_query,
                                resolution_context=resolved_context,
                            )
                            answer_body = self._strip_uncited_course_connection(llm_payload.get("answer", ""))
                            return self._chat_response(
                                answer_text=self._answer_with_source_line(answer_body, []),
                                citations=[],
                                related_kps=related,
                                mode="llm",
                                raw_query=query,
                                history=history,
                                previous_memory=session_memory,
                                resolved_query=resolved_query,
                                resolution_context=resolved_context,
                            )
                        except Exception as retry_exc:
                            print(f"[Course4186KnowledgeBase] Code retry fallback: {retry_exc}")
            return self._chat_response(
                answer_text=self._insufficient_evidence_response(resolved_query, related),
                citations=[],
                related_kps=related,
                mode="assistant",
                raw_query=query,
                history=history,
                previous_memory=session_memory,
                resolved_query=resolved_query,
                resolution_context=resolved_context,
            )

        if self._looks_brief_concept_request(resolved_query, task_profile=task_profile, kp_hits=kp_hits):
            short_answer = self._grounded_short_answer(resolved_query, citation_hits[:3], citations[:3])
            if short_answer:
                final_citations = citations[:2]
                return self._chat_response(
                    answer_text=self._answer_with_source_line(short_answer, final_citations),
                    citations=final_citations,
                    related_kps=related,
                    mode="grounded",
                    raw_query=query,
                    history=history,
                    previous_memory=session_memory,
                    resolved_query=resolved_query,
                    resolution_context=resolved_context,
                )

        if self._llm_client is not None and self._llm_model and citations:
            try:
                llm_payload = self._llm_answer(
                    query,
                    history,
                    related,
                    citation_hits[:4],
                    citations,
                    coverage_level=coverage_level,
                    task_profile=task_profile,
                    resolved_query=resolved_query,
                    resolution_context=resolved_context,
                )
                used_sources = llm_payload.get("used_sources", [])
                if task_profile.get("is_code_only_request"):
                    final_citations = []
                elif used_sources:
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
                return self._chat_response(
                    answer_text=answer,
                    citations=final_citations,
                    related_kps=related,
                    mode="llm",
                    raw_query=query,
                    history=history,
                    previous_memory=session_memory,
                    resolved_query=resolved_query,
                    resolution_context=resolved_context,
                )
            except Exception as exc:
                print(f"[Course4186KnowledgeBase] LLM answer fallback: {exc}")
                if task_profile.get("is_code_request"):
                    try:
                        llm_payload = self._llm_answer(
                            query,
                            history,
                            related,
                            [],
                            [],
                            coverage_level="none",
                            task_profile=task_profile,
                            resolved_query=resolved_query,
                            resolution_context=resolved_context,
                        )
                        answer_body = self._strip_uncited_course_connection(llm_payload.get("answer", ""))
                        return self._chat_response(
                            answer_text=self._answer_with_source_line(answer_body, []),
                            citations=[],
                            related_kps=related,
                            mode="llm",
                            raw_query=query,
                            history=history,
                            previous_memory=session_memory,
                            resolved_query=resolved_query,
                            resolution_context=resolved_context,
                        )
                    except Exception as retry_exc:
                        print(f"[Course4186KnowledgeBase] Code retry fallback: {retry_exc}")

        return self._chat_response(
            answer_text=self._answer_with_source_line(
                self._fallback_answer(resolved_query, related, citation_hits[:4], citations, task_profile=task_profile),
                citations,
            ),
            citations=citations,
            related_kps=related,
            mode="extractive",
            raw_query=query,
            history=history,
            previous_memory=session_memory,
            resolved_query=resolved_query,
            resolution_context=resolved_context,
        )

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
        cleaned = normalize_chat_math_notation(cleaned)
        cleaned = balance_markdown_code_fences(cleaned)
        cleaned = normalize_fenced_code_layout(cleaned)
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

    def _llm_generation_labels(
        self,
        query: str,
        *,
        coverage_level: str,
        task_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        if task_profile.get("is_code_only_request"):
            response_mode = "code_only"
        elif task_profile.get("is_code_plus_explanation_request"):
            response_mode = "code_plus_explanation"
        else:
            response_mode = "explanation"

        if coverage_level == "direct":
            coverage_mode = "course_direct"
        elif coverage_level == "related":
            coverage_mode = "course_related"
        else:
            coverage_mode = "course_none"

        question_mode = (
            "comparison"
            if task_profile.get("is_comparison")
            else "relation"
            if task_profile.get("is_relation")
            else "definition"
            if task_profile.get("is_definition")
            else "explanation"
        )
        lowered = self._expand_course_aliases(query)
        geometry_or_math = any(
            marker in lowered
            for marker in (
                "epipolar",
                "essential matrix",
                "fundamental matrix",
                "homography",
                "triangulation",
                "projection matrix",
                "cross product",
                "dot product",
                "camera matrix",
            )
        )
        pytorch_vision_convolution = (
            ("pytorch" in lowered or "torch" in lowered)
            and "convolution" in lowered
        )
        return {
            "response_mode": response_mode,
            "coverage_mode": coverage_mode,
            "question_mode": question_mode,
            "needs_multi_concept_coverage": bool(task_profile.get("needs_multi_concept_coverage")),
            "numeric_example_preferred": prefers_plain_numeric_math_example(query) and not task_profile.get("is_code_request"),
            "geometry_or_math_code": geometry_or_math and task_profile.get("is_code_request"),
            "pytorch_vision_convolution": pytorch_vision_convolution,
        }

    def _fallback_session_memory_update(
        self,
        raw_query: str,
        answer: str,
        previous_memory: Optional[Dict[str, Any]] = None,
        *,
        mode: str = "",
        resolved_query: str = "",
        resolution_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        previous = self._session_memory_state(previous_memory)
        effective_query = clean_display_text(resolved_query or raw_query) or clean_display_text(raw_query)
        if mode == "assistant" and not self._looks_course_request(effective_query):
            return previous

        active_topic = clean_display_text((resolution_context or {}).get("active_topic", "")) or self._derive_active_topic(
            effective_query,
            fallback=previous["active_topic"],
        )
        if not active_topic:
            active_topic = previous["active_topic"]
        return {
            "session_summary": self._compose_session_memory_summary(
                raw_query,
                effective_query,
                answer,
                previous,
                active_topic=active_topic,
            ),
            "active_topic": active_topic[:120],
        }

    def build_session_memory(
        self,
        raw_query: str,
        answer: str,
        history: Sequence[Dict[str, str]],
        previous_memory: Optional[Dict[str, Any]] = None,
        *,
        mode: str = "",
        resolved_query: str = "",
        resolution_context: Optional[Dict[str, Any]] = None,
        related: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, str]:
        previous = self._session_memory_state(previous_memory)
        effective_query = clean_display_text(resolved_query or raw_query) or clean_display_text(raw_query)
        derived_active_topic = self._derive_active_topic(
            effective_query,
            fallback=previous["active_topic"],
        )
        if mode == "assistant" and not self._looks_course_request(effective_query):
            return previous

        fallback = self._fallback_session_memory_update(
            raw_query,
            answer,
            previous,
            mode=mode,
            resolved_query=effective_query,
            resolution_context=resolution_context,
        )
        if self._llm_client is None or not self._llm_model:
            return fallback

        history_text = "\n".join(
            f"{turn.get('role', 'user')}: {clean_display_text(turn.get('content', ''))}"
            for turn in list(history)[-6:]
            if clean_display_text(turn.get("content", ""))
        )
        related_topics = "\n".join(
            f"- {clean_display_text(item.get('name', ''))}"
            for item in (related or [])
            if clean_display_text(item.get("name", ""))
        ) or "- None"
        answer_text = safe_snippet(clean_markdown_text(extract_student_answer_text(answer)), 320)
        prompt = textwrap.dedent(
            f"""
            Maintain a compact memory state for one student chat session.

            Return strict JSON only:
            {{"session_summary":"...","active_topic":"..."}}

            Rules:
            - session_summary must be 1 to 3 short sentences, under 420 characters.
            - active_topic must be a short noun phrase for the current course focus.
            - Use the latest standalone question as the main reference for topic tracking.
            - Preserve useful ongoing context for the next turn, such as whether the student wants a code example, comparison, or practice question.
            - If the latest turn clearly switched to a new in-course topic, update the topic.
            - If the latest turn is outside the course, keep the previous in-course topic instead of replacing it.
            - Do not mention file names, page numbers, source ids, retrieval, or internal system behavior.

            Previous active topic:
            {previous['active_topic'] or 'None'}

            Previous session summary:
            {previous['session_summary'] or 'None'}

            Recent conversation:
            {history_text or 'None'}

            Latest raw student message:
            {clean_display_text(raw_query) or 'None'}

            Latest standalone question:
            {effective_query or 'None'}

            Latest assistant answer summary:
            {answer_text or 'None'}

            Related course topics:
            {related_topics}
            """
        ).strip()
        try:
            response = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": "You maintain compact session memory for a course tutoring chat. Reply with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=180,
            )
            parsed = parse_json_response(response.choices[0].message.content or "")
            if not isinstance(parsed, dict):
                return fallback
            session_summary = clean_display_text(parsed.get("session_summary", ""))[:560]
            active_topic = clean_display_text(parsed.get("active_topic", ""))[:120]
            active_topic = derived_active_topic or active_topic or fallback["active_topic"]
            session_summary = self._compose_session_memory_summary(
                raw_query,
                effective_query,
                answer,
                previous,
                active_topic=active_topic,
                llm_summary=session_summary,
            )
            return {
                "session_summary": session_summary,
                "active_topic": active_topic,
            }
        except Exception:
            return fallback

    def _llm_answer_context(
        self,
        raw_query: str,
        resolved_query: str,
        history: Sequence[Dict[str, str]],
        related: Sequence[Dict[str, Any]],
        chunk_hits: Sequence[Dict[str, Any]],
        citations: Sequence[Dict[str, Any]],
        *,
        coverage_level: str,
        task_profile: Dict[str, Any],
        resolution_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        repeat_context = self._repeat_answer_context(resolved_query, history)
        repeat_count = int(repeat_context.get("repeat_count", 0) or 0)
        recent_similar_answers = list(repeat_context.get("recent_answers", []))
        recent_openings = list(repeat_context.get("recent_openings", []))
        history_text = "\n".join(
            f"{turn.get('role', 'user')}: {turn.get('content', '').strip()}"
            for turn in list(history)[-6:]
            if turn.get("content")
        )
        evidence_rows: List[str] = []
        for index, citation in enumerate(list(citations or [])[:4]):
            item = chunk_hits[index]["item"] if index < len(chunk_hits) else {}
            evidence_rows.append(
                "\n".join(
                    [
                        f"[{citation['citation_id']}] {citation.get('display_source') or citation.get('source') or ''} ({citation.get('location') or ('chunk ' + str(citation.get('chunk_index') or index + 1))})",
                        f"Section: {citation.get('section') or 'Untitled section'}",
                        f"Excerpt: {safe_snippet(item.get('text', ''), 520)}",
                    ]
                ).strip()
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
        target_concepts_text = "\n".join(
            f"- {clean_display_text(item.get('label', ''))}"
            for item in (task_profile.get("target_concepts") or [])
            if clean_display_text(item.get("label", ""))
        ) or "- None"
        session_summary = clean_display_text((resolution_context or {}).get("session_summary", ""))
        active_topic = clean_display_text((resolution_context or {}).get("active_topic", ""))
        labels = self._llm_generation_labels(
            resolved_query,
            coverage_level=coverage_level,
            task_profile=task_profile,
        )
        return {
            "query": raw_query,
            "raw_query": raw_query,
            "resolved_query": resolved_query,
            "history_text": history_text or "None",
            "evidence_text": "\n\n".join(evidence_rows) or "None",
            "kp_text": kp_text,
            "recent_answer_block": recent_answer_block,
            "recent_opening_block": recent_opening_block,
            "relevant_topics": relevant_topics,
            "target_concepts_text": target_concepts_text,
            "session_summary": session_summary,
            "active_topic": active_topic,
            "repeat_context": repeat_context,
            "repeat_count": repeat_count,
            "recent_similar_answers": recent_similar_answers,
            "recent_openings": recent_openings,
            "task_guidance": self._task_prompt_guidance(task_profile),
            "labels": labels,
            "allowed_source_ids": [
                clean_display_text(citation.get("citation_id", "")).upper()
                for citation in citations
                if clean_display_text(citation.get("citation_id", ""))
            ],
            "temperature": 0.16 if repeat_count <= 0 else min(0.42, 0.16 + 0.06 * repeat_count),
            "coverage_level": coverage_level,
            "task_profile": task_profile,
            "resolution_context": dict(resolution_context or {}),
        }

    def _llm_json_request(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        if self._llm_client is None or not self._llm_model:
            raise ValueError("LLM client is not configured.")
        response = self._llm_client.chat.completions.create(
            model=self._llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()

    def _finalize_generated_answer(
        self,
        answer_text: str,
        *,
        query: str,
        task_profile: Dict[str, Any],
    ) -> str:
        finalized = sanitize_transport_answer_text(extract_student_answer_text(answer_text))
        if not finalized:
            return ""
        finalized = promote_inline_code_blocks(finalized, query=query)
        finalized = strip_course_source_line(finalized)
        finalized = strip_citation_markers(finalized)
        finalized = normalize_answer_body_sources(finalized)
        finalized = normalize_chat_math_notation(finalized)
        if not task_profile.get("is_code_request"):
            finalized = normalize_inline_math_operators(finalized)
        finalized = balance_markdown_code_fences(finalized)
        finalized = normalize_code_fence_boundaries(finalized)
        finalized = normalize_fenced_code_layout(finalized)
        return finalized.strip()

    def _parse_llm_answer_payload(
        self,
        raw_content: str,
        *,
        query: str,
        task_profile: Dict[str, Any],
        allowed_source_ids: Sequence[str],
    ) -> Dict[str, Any]:
        allowed_source_set = {
            clean_display_text(item).upper()
            for item in allowed_source_ids
            if clean_display_text(item)
        }
        raw_used_sources = [
            source_id
            for source_id in extract_used_source_ids(raw_content)
            if source_id in allowed_source_set
        ]
        try:
            parsed = parse_json_response(raw_content)
            if isinstance(parsed, dict):
                answer_text = str(parsed.get("answer") or "").strip()
                used_sources_raw = parsed.get("used_sources", [])
                used_sources = (
                    [
                        clean_display_text(item).upper()
                        for item in used_sources_raw
                        if clean_display_text(item).upper() in allowed_source_set
                    ]
                    if isinstance(used_sources_raw, list)
                    else []
                )
                if not used_sources:
                    used_sources = raw_used_sources
                if answer_text:
                    answer_text = self._finalize_generated_answer(
                        answer_text,
                        query=query,
                        task_profile=task_profile,
                    )
                    if contains_transport_artifacts(answer_text):
                        raise ValueError("LLM answer still contained transport artifacts after JSON parse.")
                    return {
                        "answer": answer_text,
                        "used_sources": used_sources,
                    }
        except Exception:
            pass

        fallback_answer = self._finalize_generated_answer(
            raw_content,
            query=query,
            task_profile=task_profile,
        )
        if not fallback_answer:
            raise ValueError("LLM answer did not contain a usable student-facing answer.")
        if contains_transport_artifacts(fallback_answer):
            repaired = sanitize_transport_answer_text(
                normalize_answer_body_sources(
                    strip_course_source_line(
                        strip_source_id_list_suffix(fallback_answer)
                    )
                )
            )
            if repaired:
                fallback_answer = self._finalize_generated_answer(
                    repaired,
                    query=query,
                    task_profile=task_profile,
                )
        if contains_transport_artifacts(fallback_answer):
            raise ValueError("LLM answer fallback still contained transport artifacts.")
        return {
            "answer": fallback_answer,
            "used_sources": raw_used_sources,
        }

    def _answer_has_explicit_course_reference(self, text: str) -> bool:
        cleaned = normalize_answer_body_sources(extract_student_answer_text(text))
        if not cleaned:
            return False
        lowered = cleaned.lower()
        markers = (
            "course 4186",
            "this course",
            "our course",
            "the lectures",
            "lecture materials",
            "lecture ",
            "week ",
            ".pdf",
            "page ",
        )
        return any(marker in lowered for marker in markers)

    def _llm_answer_validation_issues(
        self,
        payload: Dict[str, Any],
        *,
        query: str,
        context: Dict[str, Any],
    ) -> List[str]:
        answer = str(payload.get("answer") or "").strip()
        used_sources = [
            clean_display_text(item).upper()
            for item in payload.get("used_sources", [])
            if clean_display_text(item)
        ]
        task_profile = context["task_profile"]
        labels = context["labels"]
        allowed_source_set = set(context["allowed_source_ids"])
        issues: List[str] = []

        if not answer:
            issues.append("The answer is empty.")
            return issues
        if contains_transport_artifacts(answer):
            issues.append("The answer still contains JSON, source placeholders, or transport artifacts.")
        if any(source_id not in allowed_source_set for source_id in used_sources):
            issues.append("The answer selected source ids that are not in the supplied evidence.")
        if labels["coverage_mode"] == "course_none" and used_sources:
            issues.append("The answer should not select course sources when coverage mode is course_none.")
        if not used_sources and self._answer_has_explicit_course_reference(answer):
            issues.append("The answer mentions the course or lectures explicitly without selecting a supporting source id.")
        if labels["coverage_mode"] == "course_direct" and not task_profile.get("is_code_only_request") and allowed_source_set and not used_sources:
            issues.append("The answer should retain at least one genuinely supporting course source for a directly covered topic.")
        if task_profile.get("is_code_request"):
            has_code_block = self._answer_contains_code_block(answer)
            if not has_code_block:
                issues.append("The student asked for code, but the answer does not contain a fenced Markdown code block.")
            prose_outside_code = self._strip_code_blocks_for_validation(answer)
            prose_word_count = self._meaningful_prose_word_count(prose_outside_code)
            if labels["response_mode"] == "code_only" and prose_word_count > 0:
                issues.append("This is a code-only request, so there must be no prose outside the code block.")
            if labels["response_mode"] == "code_plus_explanation" and prose_word_count < 6:
                issues.append("This request needs a brief explanation before or around the code block.")
            if has_code_block and not self._code_block_quality_ok(answer, query):
                issues.append("The code example is too thin or not complete enough to be useful.")
            issues.extend(self._code_request_topic_issues(answer, query))
        if labels["numeric_example_preferred"] and answer_uses_unfriendly_math_format(answer):
            issues.append("Use one plain-text numeric example instead of LaTeX-like or code-block math formatting.")
        if not self._answer_satisfies_task(answer, query, task_profile):
            issues.append("The answer does not fully satisfy the task implied by the student's question.")
        if context["repeat_count"] > 0 and self._answer_too_similar_to_recent(answer, context["recent_similar_answers"]):
            issues.append("The answer is too similar in wording or structure to a recent answer in the same chat.")
        return issues

    def _draft_answer(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        labels = context["labels"]
        source_ids = ", ".join(context["allowed_source_ids"]) or "None"
        system_prompt = textwrap.dedent(
            """
            You are a precise teaching assistant for a computer vision course.
            Produce the best factual draft first, then package it as JSON only.
            Return strict JSON with exactly two keys:
            - answer
            - used_sources
            Never include file names, page numbers, source ids, or source lists in the answer body.
            You may answer standard computer vision questions from reliable domain knowledge even if the lecture evidence is partial.
            Use lecture evidence only when it genuinely supports the final answer.
            For mathematical or geometric code requests, prefer a small runnable example of the key computation instead of pretending to implement a full system pipeline.
            """
        ).strip()
        user_prompt = textwrap.dedent(
            f"""
            Student question:
            {context['raw_query']}

            Standalone resolved question:
            {context['resolved_query']}

            Context resolution:
            - followup_detected: {str(context['resolution_context'].get('followup_detected', False)).lower()}
            - used_history: {str(context['resolution_context'].get('used_history', False)).lower()}
            - focus_concept: {clean_display_text(context['resolution_context'].get('focus_concept', '')) or 'None'}
            - anchor_query: {clean_display_text(context['resolution_context'].get('anchor_query', '')) or 'None'}
            - session_summary: {context['session_summary'] or 'None'}
            - active_topic: {context['active_topic'] or 'None'}

            Conversation:
            {context['history_text']}

            Variation requirement:
            {(
                "This question has appeared earlier in the same chat. Keep the facts consistent, but use a clearly different explanation path and opening sentence. "
                f"Use this variation angle: {context['repeat_context'].get('style_instruction', '')}"
            ) if context['repeat_count'] > 0 else "This is the first answer to this question in the chat, so use the clearest teaching explanation."}

            Task labels:
            - response_mode: {labels['response_mode']}
            - coverage_mode: {labels['coverage_mode']}
            - question_mode: {labels['question_mode']}
            - needs_multi_concept_coverage: {str(labels['needs_multi_concept_coverage']).lower()}
            - numeric_example_preferred: {str(labels['numeric_example_preferred']).lower()}
            - geometry_or_math_code: {str(labels['geometry_or_math_code']).lower()}
            - pytorch_vision_convolution: {str(labels['pytorch_vision_convolution']).lower()}

            Relevant course topics:
            {context['relevant_topics']}

            Target concepts to cover:
            {context['target_concepts_text']}

            Matched knowledge points:
            {context['kp_text']}

            Allowed source ids:
            {source_ids}

            Evidence:
            {context['evidence_text']}

            Drafting requirements:
            - Answer the standalone resolved question directly.
            - Preserve any output-format request from the raw student message.
            - If target concepts are listed, cover each of them explicitly and do not substitute a different concept.
            - If response_mode is code_only, put only one fenced Markdown code block in answer and nothing else.
            - If response_mode is code_plus_explanation, give a brief explanation and then one fenced Markdown code block.
            - If coverage_mode is course_direct, you may make the lecture framing central.
            - If coverage_mode is course_related, answer the concept first and keep any lecture connection brief.
            - If coverage_mode is course_none, do not mention the course or lectures and return an empty used_sources list.
            - If numeric_example_preferred is true and no code is requested, use one small numeric example in plain text, not LaTeX.
            - If pytorch_vision_convolution is true, use a 2D image example. If you call torch.nn.functional.conv2d, make both the input image and the kernel explicit 4D tensors. A nn.Conv2d example is also acceptable.
            - Keep the wording natural and student-facing, not robotic, defensive, or system-like.
            - Do not mention retrieval, grounding, provided materials, or internal rules.
            - Do not include markdown headings unless the question clearly needs them.

            Return strict JSON only.
            """
        ).strip()
        raw_content = self._llm_json_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=context["temperature"],
        )
        return self._parse_llm_answer_payload(
            raw_content,
            query=context["resolved_query"],
            task_profile=context["task_profile"],
            allowed_source_ids=context["allowed_source_ids"],
        )

    def _editorial_rewrite(
        self,
        context: Dict[str, Any],
        draft_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        labels = context["labels"]
        source_ids = ", ".join(context["allowed_source_ids"]) or "None"
        system_prompt = textwrap.dedent(
            """
            You are an editorial rewriter for a student-facing learning product.
            You may reorder, shorten, polish, or remove unsupported content, but you must not add new factual claims beyond the draft and the supplied evidence.
            Return strict JSON only with exactly two keys:
            - answer
            - used_sources
            """
        ).strip()
        user_prompt = textwrap.dedent(
            f"""
            Rewrite this draft into the final student-facing answer.

            Student question:
            {context['raw_query']}

            Standalone resolved question:
            {context['resolved_query']}

            Session memory:
            - session_summary: {context['session_summary'] or 'None'}
            - active_topic: {context['active_topic'] or 'None'}

            Variation requirement:
            {(
                "This question has already appeared in the same chat. Keep the facts stable, but do not reuse the same explanation path or opening sentence. "
                f"Use this variation angle: {context['repeat_context'].get('style_instruction', '')}"
            ) if context['repeat_count'] > 0 else "No variation constraint beyond writing naturally."}

            Task labels:
            - response_mode: {labels['response_mode']}
            - coverage_mode: {labels['coverage_mode']}
            - question_mode: {labels['question_mode']}
            - needs_multi_concept_coverage: {str(labels['needs_multi_concept_coverage']).lower()}
            - numeric_example_preferred: {str(labels['numeric_example_preferred']).lower()}
            - pytorch_vision_convolution: {str(labels['pytorch_vision_convolution']).lower()}

            Recent openings to avoid:
            {context['recent_opening_block']}

            Recent similar answers:
            {context['recent_answer_block']}

            Target concepts to preserve:
            {context['target_concepts_text']}

            Allowed source ids:
            {source_ids}

            Evidence:
            {context['evidence_text']}

            Draft JSON:
            {json.dumps(draft_payload, ensure_ascii=False)}

            Editorial rules:
            - Keep the answer natural, concise, and student-facing.
            - Do not mention system behavior, retrieval, grounding, or supplied materials.
            - Do not mention file names, page numbers, or source ids in the answer body.
            - Remove unsupported lecture claims instead of softening them.
            - Do not replace a named target concept with a different topic.
            - If response_mode is code_only, the final answer must be exactly one fenced Markdown code block and nothing else.
            - If response_mode is code_plus_explanation, the final answer must contain a brief explanation plus one fenced Markdown code block.
            - If coverage_mode is course_none, remove all lecture references and return an empty used_sources list.
            - If coverage_mode is course_direct and the answer is genuinely supported, keep at least one valid used_sources id.
            - If pytorch_vision_convolution is true, ensure the code is a valid 2D PyTorch convolution example with proper tensor shapes.
            - If the same question was answered earlier in this chat, vary the opening sentence and explanation order.

            Return strict JSON only.
            """
        ).strip()
        raw_content = self._llm_json_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=min(0.34, context["temperature"] + 0.04),
        )
        return self._parse_llm_answer_payload(
            raw_content,
            query=context["resolved_query"],
            task_profile=context["task_profile"],
            allowed_source_ids=context["allowed_source_ids"],
        )

    def _repair_answer(
        self,
        context: Dict[str, Any],
        payload: Dict[str, Any],
        issues: Sequence[str],
    ) -> Dict[str, Any]:
        labels = context["labels"]
        source_ids = ", ".join(context["allowed_source_ids"]) or "None"
        system_prompt = textwrap.dedent(
            """
            You repair student-facing answers.
            Fix only the listed issues while preserving the supported substance of the answer.
            Return strict JSON only with exactly two keys:
            - answer
            - used_sources
            """
        ).strip()
        issue_block = "\n".join(f"- {issue}" for issue in issues) or "- None"
        user_prompt = textwrap.dedent(
            f"""
            Repair the following answer.

            Student question:
            {context['raw_query']}

            Standalone resolved question:
            {context['resolved_query']}

            Session memory:
            - session_summary: {context['session_summary'] or 'None'}
            - active_topic: {context['active_topic'] or 'None'}

            Variation requirement:
            {(
                "This question has already appeared in the same chat. Fix the issues while also changing the explanation angle and opening sentence from the recent answer."
            ) if context['repeat_count'] > 0 else "No extra variation requirement."}

            Task labels:
            - response_mode: {labels['response_mode']}
            - coverage_mode: {labels['coverage_mode']}
            - question_mode: {labels['question_mode']}
            - needs_multi_concept_coverage: {str(labels['needs_multi_concept_coverage']).lower()}
            - numeric_example_preferred: {str(labels['numeric_example_preferred']).lower()}
            - pytorch_vision_convolution: {str(labels['pytorch_vision_convolution']).lower()}

            Target concepts that must still be covered:
            {context['target_concepts_text']}

            Allowed source ids:
            {source_ids}

            Evidence:
            {context['evidence_text']}

            Current JSON:
            {json.dumps(payload, ensure_ascii=False)}

            Issues to fix:
            {issue_block}

            Repair rules:
            - Keep the answer natural and student-facing.
            - Do not introduce new factual claims that are not already supported by the current answer or the evidence.
            - Do not mention file names, page numbers, source ids, or internal system behavior in the answer body.
            - Do not swap out a named target concept for a different concept.
            - If a code block is required, make it a short but complete working example.
            - If a plain-text numeric example is required, do not use LaTeX, determinant layouts, or fenced code blocks unless the student explicitly asked for code.
            - If pytorch_vision_convolution is true, keep the code as a valid 2D PyTorch convolution example with proper tensor shapes.

            Return strict JSON only.
            """
        ).strip()
        raw_content = self._llm_json_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=min(0.30, context["temperature"] + 0.02),
        )
        return self._parse_llm_answer_payload(
            raw_content,
            query=context["resolved_query"],
            task_profile=context["task_profile"],
            allowed_source_ids=context["allowed_source_ids"],
        )

    def _llm_answer(
        self,
        query: str,
        history: List[Dict[str, str]],
        related: List[Dict[str, Any]],
        chunk_hits: Sequence[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        coverage_level: str = "direct",
        task_profile: Optional[Dict[str, Any]] = None,
        resolved_query: Optional[str] = None,
        resolution_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        effective_query = clean_display_text(resolved_query or query) or clean_display_text(query)
        task_profile = task_profile or self._query_task_profile(effective_query)
        context = self._llm_answer_context(
            query,
            effective_query,
            history,
            related,
            chunk_hits,
            citations,
            coverage_level=coverage_level,
            task_profile=task_profile,
            resolution_context=resolution_context,
        )
        draft_payload = self._draft_answer(context)
        payload = self._editorial_rewrite(context, draft_payload)
        issues = self._llm_answer_validation_issues(
            payload,
            query=effective_query,
            context=context,
        )
        repair_rounds = 0
        while issues and repair_rounds < 2:
            payload = self._repair_answer(context, payload, issues)
            issues = self._llm_answer_validation_issues(
                payload,
                query=effective_query,
                context=context,
            )
            repair_rounds += 1
        if issues:
            raise ValueError("LLM answer did not satisfy the task requirements: " + " | ".join(issues))
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
