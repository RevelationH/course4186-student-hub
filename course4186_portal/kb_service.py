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
COURSE_CONTEXT_HINTS = {
    "week", "weeks", "lecture", "lectures", "tutorial", "tutorials", "revision",
    "slide", "slides", "quiz", "report", "reports", "practice", "knowledge",
    "point", "points", "topic", "topics", "material", "materials", "notes",
}
GREETING_RE = re.compile(r"^\s*(hi|hello|hey|good morning|good afternoon|good evening)\s*[!.?]*\s*$", re.IGNORECASE)
IDENTITY_RE = re.compile(r"\b(who are you|what are you|introduce yourself)\b", re.IGNORECASE)
HELP_RE = re.compile(r"\b(what can you do|how can you help|help me use this system)\b", re.IGNORECASE)
THANKS_RE = re.compile(r"\b(thanks|thank you|appreciate it)\b", re.IGNORECASE)
REPORT_RE = re.compile(r"\b(show|open|view|check).*(learning report|study summary)\b", re.IGNORECASE)
QUIZ_RE = re.compile(r"\b(open|show|start|go to).*(quiz|practice)\b", re.IGNORECASE)
DEFINITION_RE = re.compile(r"\b(what is|what are|define|definition|overview|introduce)\b", re.IGNORECASE)
CODE_REQUEST_RE = re.compile(
    r"\b("
    r"code|python|pytorch|snippet|sample code|example code|implementation|implement|write code|show code|demo code|"
    r"program|script|conv1d|conv2d|torch\.nn|torch\.nn\.functional"
    r")\b",
    re.IGNORECASE,
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
    "descriptor", "descriptors", "convolutional", "cnn",
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


def extract_student_answer_text(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if not text:
        return ""

    try:
        parsed = parse_json_response(text)
        if isinstance(parsed, dict):
            answer_text = clean_markdown_text(parsed.get("answer", ""))
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
            return clean_markdown_text(json.loads(f"\"{json_answer_match.group(1)}\""))
        except Exception:
            return clean_markdown_text(json_answer_match.group(1))

    labeled_answer_match = re.search(
        r"(?is)\banswer\s*:\s*(.+?)(?:\n\s*[\"']?used_sources[\"']?\s*:|\Z)",
        text,
    )
    if labeled_answer_match:
        candidate = labeled_answer_match.group(1).strip().strip(",")
        candidate = candidate.strip().strip("{}").strip().strip("\"'")
        return clean_markdown_text(candidate)

    cleaned = re.sub(r"(?is)(?:^|\n)\s*[\"']?used_sources[\"']?\s*:\s*\[[^\]]*\].*$", "", text).strip()
    cleaned = re.sub(r"(?is),?\s*[\"']?used_sources[\"']?\s*:\s*\[[^\]]*\]\s*\}?$", "", cleaned).strip()
    cleaned = re.sub(r"(?is)^\{\s*[\"']?answer[\"']?\s*:\s*", "", cleaned).strip()
    cleaned = re.sub(r"(?is)(?:\n|\r|\s)*Course source:.*$", "", cleaned).strip()
    cleaned = cleaned.strip().strip("{}").strip().strip("\"'")
    return clean_markdown_text(cleaned)


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

    def _content_tokens(self, query: str) -> List[str]:
        return [
            token for token in tokenize(query)
            if len(token) > 1 and token not in QUERY_NOISE_TOKENS
        ]

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
        lowered = clean_display_text(query).lower()
        tokens = self._content_tokens(query)
        return {
            "tokens": {token for token in tokens if token in self.course_anchor_tokens},
            "phrases": {phrase for phrase in self.course_anchor_phrases if phrase in lowered},
        }

    def _kp_query_match_profile(
        self,
        query: str,
        item: Dict[str, Any],
        anchor_details: Optional[Dict[str, set[str]]] = None,
    ) -> Dict[str, Any]:
        lowered = clean_display_text(query).lower()
        anchor_details = anchor_details or self._query_anchor_details(query)
        kp_name = clean_display_text(item.get("name", "")).lower()
        keyword_phrases = [
            clean_display_text(keyword).lower()
            for keyword in item.get("keywords", [])
            if clean_display_text(keyword)
        ]
        kp_terms = self.kp_term_sets.get(item.get("kp_id", ""), set())
        strong_query_tokens = anchor_details["tokens"] & STRONG_SINGLE_TERM_ANCHORS
        exact_name = bool(kp_name and kp_name in lowered)
        exact_keyword_hits = [phrase for phrase in keyword_phrases if phrase and phrase in lowered]
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
        if not anchor_details["tokens"] and not anchor_details["phrases"]:
            return False

        preferred_hits = self._preferred_chunk_hits(query, chunk_hits, limit=3, kp_context=kp_hits)
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
        return False

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

    def _focused_kp_support_hits(
        self,
        query: str,
        kp_context: Optional[Sequence[Dict[str, Any]]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        if not kp_context:
            return []
        top_hit = kp_context[0]
        top_kp = top_hit["item"]
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
        if not matched and float(top_hit.get("score", 0.0)) < 11.0:
            return []

        support_rows = self._support_candidate_rows([top_hit], limit=max(limit, 8))
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
            base_score = float(top_hit.get("score", 0.0)) + 17.0
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
                supplemental_rows = self._rerank_chunk_hits(query, supplemental_rows, [top_hit], top_k=max(limit, 8))
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
        lowered = clean_display_text(query).lower()
        tokens = self._content_tokens(query)
        if not tokens:
            return False
        if re.search(r"\b(week|lecture|tutorial|revision|slide|quiz|report|practice)\s*\d*\b", lowered):
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
        kp_hits = self.search_knowledge_points(query, top_k=1)
        if kp_hits:
            top_kp = kp_hits[0]["item"]
            kp_overlap = anchor_details["tokens"] & self.kp_term_sets.get(top_kp["kp_id"], set())
            if float(kp_hits[0]["score"]) >= 7.0 and (kp_overlap or anchor_details["phrases"]):
                return True
        if not anchor_details["tokens"] and not anchor_details["phrases"]:
            return False
        chunk_hits = self.search_chunks(query, top_k=3, kp_context=kp_hits)
        return self._is_grounded_answer_ready(query, kp_hits, chunk_hits)

    def _contextual_query(self, query: str, history: Sequence[Dict[str, str]]) -> str:
        current = clean_display_text(query)
        if not current or not history:
            return current
        lowered = current.lower()
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
            previous_lower = previous.lower()
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

    def _looks_brief_concept_request(self, query: str) -> bool:
        lowered = clean_display_text(query).lower()
        return bool(
            re.match(
                r"^\s*(what is|what are|define|definition of|introduce|explain|briefly explain|can you explain)\b",
                lowered,
            )
        )

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
        chunk_hits = self.search_chunks(retrieval_query, top_k=top_k, kp_context=kp_hits)
        focused_hits = self._focused_kp_support_hits(retrieval_query, kp_hits, limit=6)
        citation_candidates = focused_hits if len(focused_hits) >= 2 else chunk_hits
        citation_hits = self._preferred_chunk_hits(retrieval_query, citation_candidates, limit=4, kp_context=kp_hits)
        citations = self._build_citations(citation_hits, limit=4)
        related = [
            {
                "kp_id": hit["item"]["kp_id"],
                "name": hit["item"]["name"],
                "description": hit["item"]["description"],
            }
            for hit in kp_hits[:3]
        ]

        if not citations:
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
            return {
                "answer": self._insufficient_evidence_response(query, related),
                "citations": [],
                "related_kps": related,
                "mode": "assistant",
            }

        if self._looks_brief_concept_request(query):
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
                llm_payload = self._llm_answer(query, history or [], related, citation_hits[:4], citations)
                final_citations = self._select_final_citations(citations, llm_payload.get("used_sources", []), max_count=3)
                answer = self._answer_with_source_line(llm_payload.get("answer", ""), final_citations)
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
            "answer": self._fallback_answer(query, related, citation_hits[:4], citations),
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
        cleaned = extract_student_answer_text(body)
        cleaned = re.sub(r"(?:\n|\r|\s)*Course source:.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        cleaned = re.sub(r"(?:\n|\r|\s)*used_sources\s*:.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        cleaned = strip_citation_markers(cleaned)
        source_line = self._course_source_line(citations)
        if not cleaned:
            return source_line
        if not source_line:
            return cleaned
        return f"{cleaned}\n\n{source_line}"

    def _llm_answer(
        self,
        query: str,
        history: List[Dict[str, str]],
        related: List[Dict[str, Any]],
        chunk_hits: Sequence[Dict[str, Any]],
        citations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        code_request = self._query_requests_code(query)
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
        prompt = textwrap.dedent(
            f"""
            You are the Course 4186 student learning assistant for a computer vision course.
            Answer naturally, in the same language as the student's question.
            Stay strictly within the provided knowledge points and evidence.
            Do not answer from general world knowledge when the evidence is weak or unrelated.
            If the evidence does not directly define the student's term, say that clearly instead of substituting a different textbook definition.
            Do not turn a loosely related lecture mention into a full answer for a different concept.
            Do not paste raw chunk labels or say generic phrases like "I found grounded material".
            If the evidence is partial, answer only the part supported by the lectures and state the remaining limit briefly.
            Never fill missing details with textbook knowledge, common knowledge, or phrases like "well-known property".
            For definition questions, prefer the lecture's own framing over an encyclopedia-style introduction.
            For direct "What is X?" questions, paraphrase the lecture evidence as a short revision answer, not as a general encyclopedia entry.
            When the question is broad, start from the wording supported by the lecture evidence and then connect it to this course.
            Use the lecture evidence to explain the concept in a student-facing way instead of quoting slide fragments.
            Write like a teaching assistant helping a student revise this lecture before a quiz.
            Keep the answer concise, natural, and specific to the cited lecture material.
            Use 2 or 3 sentences unless the student explicitly asks for a longer derivation or comparison.
            Avoid filler phrases, repeated restatement, and broad background context that is not visible in the evidence.
            Avoid stock textbook phrases such as "fundamental concept", "various tasks", or "in the field of computer vision" unless the lecture evidence itself uses that framing.
            Avoid generic openings such as "X is a field of study", "X focuses on", "X involves techniques", or "X is a popular method" unless those exact ideas are directly supported by the evidence.
            Do not introduce extra causes, applications, or mechanisms unless they are explicitly stated in the evidence.
            If the evidence states only the core idea or constraint, stop there instead of expanding into a fuller textbook explanation.
            Do not mention lecture file names, page numbers, slide numbers, or source ids in the answer body.
            Do not include evidence labels such as [S1], [S2], or any source-id notation in the answer body.
            Do not include a source list line inside the answer body.
            {"If the student asks for code, include one short explanatory sentence before the code block and one short practical note after it. Include the code as a fenced Markdown code block with its own lines and the correct language label such as ```python. Never place multi-line code inside single backticks, and do not return code only." if code_request else ""}
            Every factual point in the answer must be supported by the selected evidence ids.
            Return strict JSON with:
            - answer: the student-facing explanation only
            - used_sources: an array of 1 to 3 ids chosen only from the evidence labels such as ["S1", "S2"]

            Conversation:
            {history_text or 'None'}

            User question:
            {query}

            Matched knowledge points:
            {kp_text}

            Evidence:
            {evidence}
            """
        ).strip()
        response = self._llm_client.chat.completions.create(
            model=self._llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You answer only from the supplied course evidence, and you should sound like a precise teaching assistant rather than a textbook or encyclopedia.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.05,
            max_tokens=420 if code_request else 220,
        )
        raw_content = (response.choices[0].message.content or "").strip()
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
                if answer_text:
                    answer_text = (
                        self._ensure_code_answer_explanation(answer_text, query, related, chunk_hits)
                        if code_request
                        else clean_markdown_text(answer_text)
                    )
                    return {
                        "answer": extract_student_answer_text(answer_text),
                        "used_sources": used_sources,
                    }
        except Exception:
            pass
        fallback_answer = extract_student_answer_text(raw_content)
        if not fallback_answer:
            raise ValueError("LLM answer did not contain a usable student-facing answer.")
        if code_request:
            fallback_answer = self._ensure_code_answer_explanation(fallback_answer, query, related, chunk_hits)
        return {
            "answer": fallback_answer,
            "used_sources": [],
        }

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
    ) -> str:
        if not citations:
            return (
                "I could not find a clear answer for that question in the current Course 4186 materials. "
                "Try using a more specific topic such as edge detection, SIFT, stereo vision, epipolar geometry, or optical flow."
            )

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
