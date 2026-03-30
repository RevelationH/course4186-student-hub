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
DISPLAY_REPLACEMENTS = {
    "\u2022": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2032": "'",
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
    "results", "applications", "draft", "appendix", "reading",
}


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


def safe_snippet(text: str, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", (text or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def clean_display_text(text: str) -> str:
    cleaned = str(text or "")
    for source, target in DISPLAY_REPLACEMENTS.items():
        cleaned = cleaned.replace(source, target)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


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
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return "/".join(parts[:-1] + [stem])


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
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


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
            if not self._is_low_priority_source(item):
                score += 5.0
            if chunk_id and chunk_id in support_chunk_ids:
                score += 6.0
            elif relative_path_key and relative_path_key in support_source_files:
                score += 3.4
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
        bonus = 0.0

        if "revision" in relative_path:
            bonus -= 3.2
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
        if "revision" in relative_path:
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
        for hit in chunk_hits[:limit]:
            item = hit["item"]
            source = clean_display_text(item.get("relative_path", ""))
            location = format_source_location(
                item.get("source_type", ""),
                item.get("unit_type", ""),
                item.get("unit_index", 0),
            )
            citations.append(
                {
                    "source": source,
                    "display_source": Path(source.replace("\\", "/")).name or source,
                    "location": location,
                    "unit_type": item.get("unit_type", ""),
                    "unit_index": item.get("unit_index", 0),
                    "chunk_index": item.get("chunk_index", 0),
                    "section": clean_display_text(item.get("title", "")) or location,
                    "snippet": safe_snippet(item.get("text", ""), 220),
                }
            )
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
        query_lower = clean_display_text(query).lower()
        kp_name = clean_display_text(top_kp.get("name", "")).lower()
        kp_keywords = [clean_display_text(keyword).lower() for keyword in top_kp.get("keywords", [])]
        matched = bool(kp_name and kp_name in query_lower) or any(keyword and keyword in query_lower for keyword in kp_keywords)
        if not matched and float(top_hit.get("score", 0.0)) < 11.0:
            return []

        support_rows = self._support_candidate_rows([top_hit], limit=max(limit, 8))
        if len(support_rows) < 2:
            return []

        anchor_details = self._query_anchor_details(query)
        focused = [
            row for row in support_rows
            if self._phrase_overlap_with_item(row["item"], anchor_details)
            or self._anchor_overlap_with_item(row["item"], anchor_details) >= 1
        ]
        if len(focused) >= 2:
            return focused[:limit]
        return support_rows[:limit] if matched else []

    def _rerank_kp_hits(self, query: str, hits: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        query_terms = set(self._content_tokens(query))
        ranking_query = self._ranking_query(query).lower()
        reranked: List[Dict[str, Any]] = []
        for hit in hits:
            item = hit["item"]
            kp_terms = self.kp_term_sets.get(item["kp_id"], set())
            overlap = len(query_terms & kp_terms)
            phrase_bonus = 3.0 if ranking_query and ranking_query in " ".join(sorted(kp_terms)) else 0.0
            score = float(hit["score"]) + overlap * 4.0 + phrase_bonus
            if overlap or phrase_bonus or self._course_term_matches(query_terms):
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
            phrase_bonus = 4.0 if ranking_query and ranking_query in blob else 0.0
            source_bonus = 0.0
            if kp_weeks:
                source_bonus += 1.2 if str(item.get("week") or "") in kp_weeks else -0.8
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
        explanation = clean_display_text(item.get("explanation", ""))
        if not question or not explanation:
            return None

        raw_options = item.get("options") or []
        if not isinstance(raw_options, list) or len(raw_options) != 4:
            return None

        normalized_options: List[str] = []
        for expected_label, raw_option in zip("ABCD", raw_options):
            option_text = clean_display_text(str(raw_option or ""))
            if not option_text:
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
        seen: set[tuple[str, str, str]] = set()
        for ref in review_refs:
            source = clean_display_text(ref.get("source", ""))
            location = clean_display_text(ref.get("location", ""))
            section = clean_display_text(ref.get("section", ""))
            key = (source.lower(), location.lower(), section.lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(ref))
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

    def _assistant_response_for_query(self, query: str) -> Optional[str]:
        intent = self._assistant_intent(query)
        if not intent:
            return None

        if self._llm_client is not None and self._llm_model:
            try:
                return self._llm_assistant_response(query, intent)
            except Exception:
                pass

        if intent == "greeting":
            return (
                "Hello. I am the Course 4186 student learning assistant. "
                "You can ask me about lecture topics, formulas, or quiz practice."
            )
        if intent == "identity":
            return (
                "I am the Course 4186 student learning assistant. "
                "I help students answer lecture-based questions, review knowledge points, and move into quiz practice."
            )
        if intent == "help":
            return (
                "I can help with Course 4186 lecture content, explain knowledge points, and guide you to Chat or Quiz practice. "
                "Try asking about computer vision, edge detection, SIFT, camera geometry, or image alignment."
            )
        if intent == "report":
            return (
                "This student version focuses on Chat and Quiz only. "
                "If you want to review progress, use Quiz practice and the dashboard summary."
            )
        if intent == "quiz":
            return (
                "You can open Quiz from the left sidebar. "
                "Practice is organized by knowledge point, and each set uses multiple-choice questions."
            )
        if intent == "thanks":
            return "You are welcome. If you want, ask me about any Course 4186 topic or open Quiz for more practice."
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

    def _out_of_scope_response(self, query: str) -> str:
        return (
            "I can help only with Course 4186 topics. "
            "Ask about computer vision concepts such as convolution, SIFT, camera geometry, stereo vision, or optical flow."
        )

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
            - Write in natural exam style, not chatbot style.
            - Prefer concrete concept checks, reasoning steps, equation interpretation, method choice, error diagnosis, or small calculation-style prompts when the evidence supports them.
            - Avoid weak prompts such as "Which keyword...", "Which lecture...", "Which topic matches...", or anything asking about file names, week numbers, or slide numbers.
            - Avoid simply paraphrasing the existing fixed-bank questions.
            - Keep each question self-contained and answerable from the provided evidence.
            - The explanation must briefly justify the correct answer and connect back to the lecture idea the student needs to review.
            - Do not invent facts beyond the evidence.
            """
        ).strip()

        try:
            response = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Generate grounded exam-style multiple-choice questions only. Output strict JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.25,
                max_tokens=max(480, question_count * 320),
            )
            parsed = parse_json_response(response.choices[0].message.content or "[]")
        except Exception as exc:
            print(f"[Course4186KnowledgeBase] Follow-up variant generation fallback: {exc}")
            return []

        if isinstance(parsed, dict):
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
        special_response = self._assistant_response_for_query(query)
        if special_response:
            return {
                "answer": special_response,
                "citations": [],
                "related_kps": [],
                "mode": "assistant",
            }

        if not self._looks_course_request(query):
            return {
                "answer": self._out_of_scope_response(query),
                "citations": [],
                "related_kps": [],
                "mode": "assistant",
            }

        kp_hits = self.search_knowledge_points(query, top_k=3)
        chunk_hits = self.search_chunks(query, top_k=top_k, kp_context=kp_hits)
        focused_hits = self._focused_kp_support_hits(query, kp_hits, limit=6)
        citation_candidates = focused_hits if len(focused_hits) >= 2 else chunk_hits
        citation_hits = self._preferred_chunk_hits(query, citation_candidates, limit=4, kp_context=kp_hits)
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

        if not self._is_grounded_answer_ready(query, kp_hits, citation_hits):
            return {
                "answer": self._insufficient_evidence_response(query, related),
                "citations": [],
                "related_kps": related,
                "mode": "assistant",
            }

        if self._llm_client is not None and self._llm_model and citations:
            try:
                answer = self._llm_answer(query, history or [], related, citation_hits[:4], citations)
                return {
                    "answer": answer,
                    "citations": citations,
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
            "display_source": Path(clean_display_text(primary.get("relative_path", "")).replace("\\", "/")).name,
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

    def _llm_answer(
        self,
        query: str,
        history: List[Dict[str, str]],
        related: List[Dict[str, Any]],
        chunk_hits: Sequence[Dict[str, Any]],
        citations: List[Dict[str, Any]],
    ) -> str:
        history_text = "\n".join(
            f"{turn.get('role', 'user')}: {turn.get('content', '').strip()}"
            for turn in history[-6:]
            if turn.get("content")
        )
        evidence = "\n\n".join(
            (
                f"Source: {citation['source']} ({citation.get('location') or ('chunk ' + str(citation['chunk_index']))})\n"
                f"Section: {citation.get('section') or 'Untitled section'}\n"
                f"Excerpt: {safe_snippet(hit['item'].get('text', ''), 520)}"
            )
            for hit, citation in zip(chunk_hits, citations)
        )
        kp_text = "\n".join(f"- {item['name']}: {item['description']}" for item in related) or "- None"
        prompt = textwrap.dedent(
            f"""
            You are the Course 4186 student learning assistant for a computer vision course.
            Answer naturally, in the same language as the student's question.
            Stay strictly within the provided knowledge points and evidence.
            Do not answer from general world knowledge when the evidence is weak or unrelated.
            If the evidence does not directly define the student's term, say that clearly instead of substituting a different textbook definition.
            Do not turn a loosely related lecture mention into a full answer for a different concept.
            Do not paste raw chunk labels or say generic phrases like "I found grounded material".
            If the evidence is partial, give the supported answer first and then note the limit briefly.
            When the question is broad, begin with a plain-language definition and then connect it to this course.
            Use the lecture evidence to explain the concept in a student-facing way instead of quoting slide fragments.
            Mention the most relevant file name and page or slide number in the answer.
            End with one short line starting with "Course source:" followed by 1 to 3 file-and-location references.
            Keep the answer concise but conversational.

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
                    "content": "You answer only from the supplied course evidence, but you should sound like a normal teaching assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=700,
        )
        return (response.choices[0].message.content or "").strip()

    def _llm_assistant_response(self, query: str, intent: str) -> str:
        intent_guidance = {
            "greeting": "Greet the student naturally and invite a course-related question.",
            "identity": "Explain who you are in one or two plain, professional sentences.",
            "help": "Explain briefly what kinds of Course 4186 help you can provide.",
            "report": "Explain that this student version focuses on Chat and Quiz rather than a separate learning-report page.",
            "quiz": "Tell the student how to use quiz practice in a natural way.",
            "thanks": "Reply politely and invite the next course-related question.",
            "out_of_scope": "Politely say you focus on Course 4186, do not answer the unrelated question itself, and redirect to nearby course topics in a natural way.",
        }
        prompt = textwrap.dedent(
            f"""
            You are the Course 4186 student learning assistant.
            Reply in the same language as the student's message.
            Keep the reply natural, short, student-facing, and professionally neutral.
            Avoid cheerleading, jokes, emojis, and overly enthusiastic phrases.
            Do not mention internal rules, retrieval, grounding, or system behavior.
            Do not answer unrelated factual questions outside Course 4186.

            Interaction type:
            {intent}

            Guidance:
            {intent_guidance.get(intent, "Reply naturally and stay within the Course 4186 assistant role.")}

            Student message:
            {clean_display_text(query)}
            """
        ).strip()
        response = self._llm_client.chat.completions.create(
            model=self._llm_model,
            messages=[
                {"role": "system", "content": "You are a concise course assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.35,
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
        source_ref = f"{citations[0].get('display_source') or citations[0]['source']} ({citations[0].get('location')})"
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

        extra_sources = [
            f"{item.get('display_source') or item['source']} ({item.get('location')})"
            for item in citations[1:3]
        ]
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
