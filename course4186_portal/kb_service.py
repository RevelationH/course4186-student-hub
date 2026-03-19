from __future__ import annotations

import json
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
    "course", "would", "could", "should", "into", "only", "over",
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
            idf = __import__("math").log(1 + (doc_count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
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
        self.questions_by_kp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for question in self.questions:
            enriched = dict(question)
            enriched["parsed_options"] = parse_options(question.get("options", []))
            self.questions_by_kp[question["kp_id"]].append(enriched)
        self.course_terms = self._build_course_terms()

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

    def _content_tokens(self, query: str) -> List[str]:
        return [
            token for token in tokenize(query)
            if len(token) > 1 and token not in QUERY_NOISE_TOKENS
        ]

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
                "You can ask me about lecture topics, formulas, quiz practice, or your learning report."
            )
        if intent == "identity":
            return (
                "I am the Course 4186 student learning assistant. "
                "I help students answer lecture-based questions, review knowledge points, and move into quiz practice or learning reports."
            )
        if intent == "help":
            return (
                "I can help with Course 4186 lecture content, explain knowledge points, and guide you to quiz practice or your learning report. "
                "Try asking about computer vision, edge detection, SIFT, camera geometry, or image alignment."
            )
        if intent == "report":
            return (
                "You can open your Learning Report from the left sidebar. "
                "It summarizes your recent answers, weak areas, and recommended next steps."
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
        if "computer vision" in lowered:
            return True
        if re.search(r"\b(week|lecture|tutorial|revision|slide|quiz|report|practice)\s*\d*\b", lowered):
            return True
        if any(token in self.course_terms for token in tokens):
            return True
        return bool(self.search_knowledge_points(query, top_k=1))

    def _out_of_scope_response(self, query: str) -> str:
        if self._llm_client is not None and self._llm_model:
            try:
                return self._llm_assistant_response(query, "out_of_scope")
            except Exception:
                pass
        return (
            "I focus on Course 4186, so I cannot help much with unrelated topics. "
            "If you want, ask about computer vision foundations, filtering, edges, CNNs, SIFT, or image alignment instead."
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
        if not api_key:
            return
        base_url = os.getenv("COURSE_LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("KIMI_BASE_URL")
        model = os.getenv("COURSE_LLM_MODEL") or os.getenv("DEEPSEEK_CHAT_MODEL") or os.getenv("KIMI_MODEL") or "deepseek-chat"
        try:
            self._llm_client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
            self._llm_model = model
        except Exception:
            self._llm_client = None
            self._llm_model = None

    @property
    def dense_enabled(self) -> bool:
        return self._dense_enabled

    def list_knowledge_points(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for kp in self.kps:
            rows.append(
                {
                    "kp_id": kp["kp_id"],
                    "name": kp["name"],
                    "description": kp["description"],
                    "weeks": kp.get("weeks", []),
                    "question_count": len(self.questions_by_kp.get(kp["kp_id"], [])),
                }
            )
        rows.sort(key=lambda row: (row["weeks"][0] if row["weeks"] else "ZZZ", row["name"]))
        return rows

    def get_kp(self, kp_id: str) -> Optional[Dict[str, Any]]:
        kp = self.kp_by_id.get(kp_id)
        if not kp:
            return None
        payload = dict(kp)
        payload["questions"] = list(self.questions_by_kp.get(kp_id, []))
        return payload

    def related_questions(self, kp_id: str) -> List[Dict[str, Any]]:
        return list(self.questions_by_kp.get(kp_id, []))

    def search_knowledge_points(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        hits = lexical_rank(
            query,
            self.kps,
            text_getter=lambda kp: "\n".join([kp["name"], kp["description"], " ".join(kp.get("keywords", []))]),
            top_k=top_k,
        )
        return [{"score": round(score, 3), "item": item} for score, item in hits]

    def search_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        lexical_hits = lexical_rank(
            query,
            self.chunks,
            text_getter=lambda chunk: "\n".join([chunk["title"], chunk["relative_path"], chunk["text"]]),
            top_k=top_k,
        )
        lexical_rows = [{"score": round(score, 3), "item": item} for score, item in lexical_hits]

        if len(self._content_tokens(query)) <= 3 and lexical_rows:
            return lexical_rows

        if self._dense_enabled and self._vector_store is not None:
            try:
                dense_hits = self._vector_store.similarity_search_with_score(query, k=top_k)
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
                    for hit in rows + lexical_rows:
                        chunk_id = hit["item"].get("chunk_id")
                        if chunk_id in seen_chunk_ids:
                            continue
                        seen_chunk_ids.add(chunk_id)
                        merged.append(hit)
                        if len(merged) >= top_k:
                            break
                    return merged
            except Exception:
                pass

        return lexical_rows

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
        chunk_hits = self.search_chunks(query, top_k=top_k)
        citations = [
            {
                "source": hit["item"]["relative_path"],
                "location": format_source_location(
                    hit["item"].get("source_type", ""),
                    hit["item"].get("unit_type", ""),
                    hit["item"].get("unit_index", 0),
                ),
                "unit_type": hit["item"].get("unit_type", ""),
                "unit_index": hit["item"].get("unit_index", 0),
                "chunk_index": hit["item"]["chunk_index"],
                "snippet": safe_snippet(hit["item"]["text"], 180),
            }
            for hit in chunk_hits[:4]
        ]
        related = [
            {
                "kp_id": hit["item"]["kp_id"],
                "name": hit["item"]["name"],
                "description": hit["item"]["description"],
            }
            for hit in kp_hits[:3]
        ]

        if self._llm_client is not None and self._llm_model and citations:
            try:
                answer = self._llm_answer(query, history or [], related, citations)
                return {
                    "answer": answer,
                    "citations": citations,
                    "related_kps": related,
                    "mode": "llm",
                }
            except Exception:
                pass

        return {
            "answer": self._fallback_answer(query, related, citations),
            "citations": citations,
            "related_kps": related,
            "mode": "extractive",
        }

    def _llm_answer(
        self,
        query: str,
        history: List[Dict[str, str]],
        related: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
    ) -> str:
        history_text = "\n".join(
            f"{turn.get('role', 'user')}: {turn.get('content', '').strip()}"
            for turn in history[-6:]
            if turn.get("content")
        )
        evidence = "\n\n".join(
            f"[{item['source']} | {item.get('location') or ('chunk ' + str(item['chunk_index']))}] {item['snippet']}"
            for item in citations
        )
        kp_text = "\n".join(f"- {item['name']}: {item['description']}" for item in related) or "- None"
        prompt = textwrap.dedent(
            f"""
            You are the Course 4186 student learning assistant for a computer vision course.
            Answer naturally, in the same language as the student's question.
            Stay strictly within the provided knowledge points and evidence.
            Do not paste raw chunk labels or say generic phrases like "I found grounded material".
            If the evidence is partial, give the supported answer first and then note the limit briefly.
            When the question is broad, begin with a plain-language definition and then connect it to this course.
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
            "report": "Tell the student how to use the learning report in a natural way.",
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
        citations: List[Dict[str, Any]],
    ) -> str:
        if not citations:
            return (
                "I could not find grounded material for that question in the Course 4186 knowledge base. "
                "Try using a more specific topic such as edge detection, SIFT, stereo vision, epipolar geometry, or optical flow."
            )

        top_kp = related[0] if related else None
        lines: List[str] = []
        if top_kp:
            if "use" in query.lower() or "used for" in query.lower() or "why" in query.lower():
                lines.append(f"{top_kp['name']} is mainly used for {top_kp['description'].rstrip('.').lower()}.")
            else:
                lines.append(f"In Course 4186, {top_kp['name']} refers to {top_kp['description'].rstrip('.').lower()}.")
        else:
            lines.append("Here are the most relevant notes I found in the course materials.")

        lines.append("Evidence from the course materials:")
        for item in citations[:3]:
            location = item.get("location") or f"chunk {item['chunk_index']}"
            lines.append(f"- {item['snippet']} [{item['source']} | {location}]")

        if len(related) > 1:
            others = ", ".join(item["name"] for item in related[1:3])
            lines.append(f"Related topics: {others}.")
        return "\n".join(lines)

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
