from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import posixpath
import random
import re
import shutil
import sys
import textwrap
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

# Keep the local RAG pipeline on the PyTorch path so it does not pick up
# unrelated TensorFlow/Keras packages installed in the base environment.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pypdf import PdfReader
try:
    from question_blueprints import QUESTION_BLUEPRINTS
except Exception:
    from exam_question_blueprints import QUESTION_BLUEPRINTS

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from ftfy import fix_text as ftfy_fix_text
except Exception:
    ftfy_fix_text = None

try:
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except Exception:
    Document = None
    RecursiveCharacterTextSplitter = None
    HuggingFaceEmbeddings = None
    FAISS = None

from env_loader import load_project_env

load_project_env()


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COURSE_ROOT = Path(r"D:\digital_human\4186\4186")
DEFAULT_ARTIFACT_DIR = SCRIPT_DIR / "artifacts"
DEFAULT_EMBED_MODEL = os.getenv("COURSE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SUPPORTED_EXTENSIONS = {".pdf"}
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*")
WS_RE = re.compile(r"\s+")
WEEK_LABEL_RE = re.compile(r"^Week\s*(\d+)$", re.IGNORECASE)
NOISE_PATTERNS = [
    re.compile(r"^Ignoring wrong pointing object", re.IGNORECASE),
]
PPTX_NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}
PPT_REL_NS = {
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
DOCX_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
BROWSER_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
MIN_SLIDE_IMAGE_BYTES = 4096
MIN_QUESTION_COUNT = 5
SOURCE_TYPE_PRIORITY = {"pdf": 0, "pptx": 1, "ppt": 1, "docx": 2}
REVIEW_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "which", "when",
    "where", "how", "why", "into", "onto", "about", "under", "over", "their", "there",
    "these", "those", "used", "using", "being", "between", "through", "after", "before",
    "during", "main", "best", "most", "does", "used", "same", "course", "lecture",
    "question", "image", "figure", "shown", "showing", "illustrating", "system", "task",
    "point", "points", "would", "should", "could", "because", "into", "only", "than",
    "then", "while", "they", "them", "their", "your", "yours", "have", "has", "had",
    "are", "was", "were", "is", "to", "of", "in", "on", "at", "by", "as", "be", "a",
    "an", "or", "it", "if",
}
QUIZ_EXCLUDED_KP_NAMES = {
    "LLM prompting and DeepSeek overview",
}


@dataclass
class RawDocument:
    doc_id: str
    relative_path: str
    source_type: str
    week: str
    unit_type: str
    unit_index: int
    title: str
    text: str


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    relative_path: str
    source_type: str
    week: str
    title: str
    unit_type: str
    unit_index: int
    chunk_index: int
    text: str


@dataclass
class KnowledgePoint:
    kp_id: str
    name: str
    description: str
    keywords: List[str] = field(default_factory=list)
    weeks: List[str] = field(default_factory=list)
    source_files: List[str] = field(default_factory=list)
    support_chunk_ids: List[str] = field(default_factory=list)
    support_preview: List[str] = field(default_factory=list)


@dataclass
class QuestionRecord:
    question_id: str
    kp_id: str
    kp_name: str
    question_type: str
    question: str
    answer: str
    explanation: str
    options: List[str] = field(default_factory=list)
    correct_option: Optional[str] = None
    source_chunk_ids: List[str] = field(default_factory=list)
    source_files: List[str] = field(default_factory=list)
    review_refs: List[Dict[str, Any]] = field(default_factory=list)
    image_path: Optional[str] = None
    image_caption: Optional[str] = None


@dataclass
class SlideImageRecord:
    doc_id: str
    relative_path: str
    slide_index: int
    image_path: str
    image_caption: str
    image_name: str
    size_bytes: int


def short_hash(text: str, length: int = 12) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or short_hash(text)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def emit_output(text: str) -> None:
    sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))


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


def normalize_text(text: str) -> str:
    lines: List[str] = []
    repaired = repair_text_encoding(text).replace("\x00", " ")
    for raw_line in repaired.splitlines():
        line = raw_line
        for source, target in DISPLAY_REPLACEMENTS.items():
            line = line.replace(source, target)
        line = WS_RE.sub(" ", line).strip()
        if not line:
            continue
        if any(pattern.search(line) for pattern in NOISE_PATTERNS):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def guess_title(text: str, fallback: str) -> str:
    for line in text.splitlines()[:8]:
        line = line.strip(" -:\t")
        if 3 <= len(line) <= 120:
            return line
    return fallback


def detect_week(relative_path: Path) -> str:
    parts = relative_path.parts
    if not parts:
        return "Unknown"
    if parts[0].lower() == "tutorial" and len(parts) > 1:
        return f"Tutorial/{parts[1]}"
    return parts[0]


def parse_week_number(label: str) -> Optional[int]:
    text = str(label or "").strip()
    if not text:
        return None
    if "/" in text:
        text = text.split("/")[-1]
    match = WEEK_LABEL_RE.match(text)
    return int(match.group(1)) if match else None


def within_max_week(relative_path: Path, max_week: Optional[int]) -> bool:
    if max_week is None:
        return True
    week_label = detect_week(relative_path)
    week_number = parse_week_number(week_label)
    if week_number is None:
        return False
    return week_number <= max_week


def iter_course_files(course_root: Path, max_week: Optional[int] = None) -> List[Path]:
    files = [
        path for path in course_root.rglob("*")
        if path.is_file() and path.name != ".DS_Store" and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if max_week is not None:
        files = [path for path in files if within_max_week(path.relative_to(course_root), max_week)]
    return sorted(files)


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def safe_snippet(text: str, limit: int = 180) -> str:
    compact = WS_RE.sub(" ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


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
    "\u03bb": "lambda",
    "\u00a0": " ",
    "\uf06c": "lambda",
    "\uf0b7": "-",
    "\ufffd": "",
    "\u0ddc": "",
    "\u0ddd": "",
    "\U0001d465": "x",
}


def clean_display_text(text: str) -> str:
    cleaned = repair_text_encoding(text)
    for source, target in DISPLAY_REPLACEMENTS.items():
        cleaned = cleaned.replace(source, target)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def canonical_display_source_name(relative_path: str) -> str:
    cleaned = clean_display_text(relative_path).replace("\\", "/")
    name = PurePosixPath(cleaned).name or cleaned
    stem = strip_trailing_copy_markers(PurePosixPath(name).stem)
    suffix = PurePosixPath(name).suffix.lower()
    if suffix in {".ppt", ".pptx", ".doc", ".docx"}:
        suffix = ".pdf"
    return f"{stem}{suffix}" if stem and suffix else (stem or name)


def material_stem_key(name: str) -> str:
    cleaned = strip_trailing_copy_markers(name).lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return " ".join(cleaned.split())


def find_pptx_companion(course_root: Path, relative_path: Path) -> Optional[Path]:
    candidate_dir = course_root / relative_path.parent
    if not candidate_dir.exists() or not candidate_dir.is_dir():
        return None
    source_key = material_stem_key(relative_path.stem)
    if not source_key:
        return None
    exact = candidate_dir / f"{relative_path.stem}.pptx"
    if exact.exists():
        return exact.relative_to(course_root)
    for candidate in sorted(candidate_dir.glob("*.pptx")):
        if material_stem_key(candidate.stem) == source_key:
            return candidate.relative_to(course_root)
    return None


def display_unit_type(display_source: str, unit_type: str) -> str:
    suffix = PurePosixPath(clean_display_text(display_source or "")).suffix.lower()
    normalized_unit_type = str(unit_type or "").lower()
    if suffix == ".pdf" and normalized_unit_type == "slide":
        return "page"
    return normalized_unit_type


def source_unit_family_key(relative_path: str, unit_type: str, unit_index: Any) -> Tuple[str, str, int]:
    try:
        normalized_index = int(unit_index or 0)
    except Exception:
        normalized_index = 0
    path = clean_display_text(relative_path).replace("\\", "/")
    parent = str(PurePosixPath(path).parent)
    stem = strip_trailing_copy_markers(PurePosixPath(path).stem).lower()
    return (f"{parent}/{stem}".strip("/"), str(unit_type or "").lower(), normalized_index)


def source_family_key(relative_path: str) -> str:
    path = clean_display_text(relative_path).replace("\\", "/")
    parent = str(PurePosixPath(path).parent)
    stem = strip_trailing_copy_markers(PurePosixPath(path).stem).lower()
    return f"{parent}/{stem}".strip("/")


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
    relative_path: str,
    source_type: str,
    unit_type: str,
    unit_index: Any,
    display_source: str = "",
) -> str:
    resolved_display_source = canonical_display_source_name(display_source or relative_path)
    resolved_unit_type = display_unit_type(resolved_display_source, unit_type)
    resolved_source_type = "pdf" if resolved_unit_type == "page" else source_type
    return format_source_location(resolved_source_type, resolved_unit_type, unit_index)


def sort_slide_name(name: str) -> Tuple[int, str]:
    match = re.search(r"slide(\d+)\.xml$", name)
    return (int(match.group(1)) if match else 0, name)


def parse_docx_paragraphs(blob: bytes) -> List[str]:
    root = ET.fromstring(blob)
    paragraphs: List[str] = []
    for paragraph in root.findall(".//w:p", DOCX_NS):
        parts = [node.text for node in paragraph.findall(".//w:t", DOCX_NS) if node.text]
        text = normalize_text("".join(parts))
        if text:
            paragraphs.append(text)
    return paragraphs


def extract_pdf_units(file_path: Path, relative_path: Path) -> List[RawDocument]:
    docs: List[RawDocument] = []
    reader = PdfReader(str(file_path))
    week = detect_week(relative_path)
    for page_index, page in enumerate(reader.pages, start=1):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            extracted = page.extract_text() or ""
        text = normalize_text(extracted)
        if not text:
            continue
        docs.append(
            RawDocument(
                doc_id=f"raw-{short_hash(f'{relative_path}|page|{page_index}')}",
                relative_path=str(relative_path),
                source_type="pdf",
                week=week,
                unit_type="page",
                unit_index=page_index,
                title=guess_title(text, file_path.stem),
                text=text,
            )
        )
    return docs


def extract_pptx_units(file_path: Path, relative_path: Path) -> List[RawDocument]:
    docs: List[RawDocument] = []
    week = detect_week(relative_path)
    with zipfile.ZipFile(file_path) as archive:
        slide_names = [
            name for name in archive.namelist()
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        ]
        for slide_name in sorted(slide_names, key=sort_slide_name):
            slide_index = sort_slide_name(slide_name)[0]
            root = ET.fromstring(archive.read(slide_name))
            text = normalize_text(" ".join(node.text for node in root.findall(".//a:t", PPTX_NS) if node.text))
            if not text:
                continue
            docs.append(
                RawDocument(
                    doc_id=f"raw-{short_hash(f'{relative_path}|slide|{slide_index}')}",
                    relative_path=str(relative_path),
                    source_type="pptx",
                    week=week,
                    unit_type="slide",
                    unit_index=slide_index,
                    title=guess_title(text, file_path.stem),
                    text=text,
                )
            )
    return docs


def extract_docx_units(file_path: Path, relative_path: Path) -> List[RawDocument]:
    week = detect_week(relative_path)
    with zipfile.ZipFile(file_path) as archive:
        paragraphs = parse_docx_paragraphs(archive.read("word/document.xml"))
    text = normalize_text("\n".join(paragraphs))
    if not text:
        return []
    return [
        RawDocument(
            doc_id=f"raw-{short_hash(f'{relative_path}|document|1')}",
            relative_path=str(relative_path),
            source_type="docx",
            week=week,
            unit_type="document",
            unit_index=1,
            title=guess_title(text, file_path.stem),
            text=text,
        )
    ]


def extract_ppt_units(file_path: Path, relative_path: Path) -> List[RawDocument]:
    try:
        import pythoncom
        import win32com.client
    except Exception as exc:
        raise RuntimeError(f"PowerPoint COM is unavailable: {exc}") from exc

    docs: List[RawDocument] = []
    week = detect_week(relative_path)
    app = None
    presentation = None
    pythoncom.CoInitialize()
    try:
        app = win32com.client.Dispatch("PowerPoint.Application")
        presentation = app.Presentations.Open(str(file_path), False, False, False)
        for slide_number in range(1, presentation.Slides.Count + 1):
            slide = presentation.Slides(slide_number)
            fragments: List[str] = []
            for shape in slide.Shapes:
                try:
                    if shape.HasTextFrame and shape.TextFrame.HasText:
                        fragments.append(str(shape.TextFrame.TextRange.Text))
                except Exception:
                    continue
            text = normalize_text("\n".join(fragments))
            if not text:
                continue
            docs.append(
                RawDocument(
                    doc_id=f"raw-{short_hash(f'{relative_path}|slide|{slide_number}')}",
                    relative_path=str(relative_path),
                    source_type="ppt",
                    week=week,
                    unit_type="slide",
                    unit_index=slide_number,
                    title=guess_title(text, file_path.stem),
                    text=text,
                )
            )
    finally:
        if presentation is not None:
            try:
                presentation.Close()
            except Exception:
                pass
        if app is not None:
            try:
                app.Quit()
            except Exception:
                pass
        pythoncom.CoUninitialize()
    return docs


def extract_raw_documents(course_root: Path, max_week: Optional[int] = None) -> Tuple[List[RawDocument], List[Dict[str, str]], List[Path]]:
    raw_documents: List[RawDocument] = []
    issues: List[Dict[str, str]] = []
    course_files = iter_course_files(course_root, max_week=max_week)
    for file_path in course_files:
        relative_path = file_path.relative_to(course_root)
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            issues.append(
                {
                    "relative_path": str(relative_path),
                    "issue": f"Unsupported extension: {suffix}",
                }
            )
            continue
        try:
            if suffix == ".pdf":
                docs = extract_pdf_units(file_path, relative_path)
            elif suffix == ".pptx":
                docs = extract_pptx_units(file_path, relative_path)
            elif suffix == ".docx":
                docs = extract_docx_units(file_path, relative_path)
            else:
                docs = extract_ppt_units(file_path, relative_path)
            raw_documents.extend(docs)
            if not docs:
                issues.append(
                    {
                        "relative_path": str(relative_path),
                        "issue": "No extractable text found",
                    }
                )
        except Exception as exc:
            issues.append(
                {
                    "relative_path": str(relative_path),
                    "issue": f"{type(exc).__name__}: {exc}",
                }
            )
    return raw_documents, issues, course_files


def extract_slide_images_for_docs(
    course_root: Path,
    artifact_dir: Path,
    raw_documents: Sequence[RawDocument],
) -> Tuple[List[SlideImageRecord], List[Dict[str, str]]]:
    image_root = artifact_dir / "images"
    if image_root.exists():
        shutil.rmtree(image_root)
    ensure_dir(image_root)

    docs_by_file: Dict[str, Dict[int, RawDocument]] = defaultdict(dict)
    for doc in raw_documents:
        if doc.source_type == "pptx" and doc.unit_type == "slide":
            docs_by_file[doc.relative_path][doc.unit_index] = doc
            continue
        if doc.source_type == "pdf" and doc.unit_type == "page":
            companion = find_pptx_companion(course_root, Path(doc.relative_path))
            if companion is not None:
                docs_by_file[str(companion)][doc.unit_index] = doc

    records: List[SlideImageRecord] = []
    issues: List[Dict[str, str]] = []
    for image_source_relative_path, slides in sorted(docs_by_file.items()):
        file_path = course_root / Path(image_source_relative_path)
        if not file_path.exists():
            issues.append({"relative_path": image_source_relative_path, "issue": "PPTX file not found during image extraction"})
            continue
        try:
            with zipfile.ZipFile(file_path) as archive:
                for slide_index, doc in sorted(slides.items()):
                    slide_name = f"ppt/slides/slide{slide_index}.xml"
                    rel_name = f"ppt/slides/_rels/slide{slide_index}.xml.rels"
                    if rel_name not in archive.namelist():
                        continue

                    candidates: List[Tuple[int, str, bytes]] = []
                    root = ET.fromstring(archive.read(rel_name))
                    for relationship in root.findall(".//rel:Relationship", PPT_REL_NS):
                        target = (relationship.attrib.get("Target") or "").replace("\\", "/")
                        if "/media/" not in target and not target.startswith("../media/"):
                            continue

                        source_dir = PurePosixPath(slide_name).parent.as_posix()
                        archive_member = posixpath.normpath(posixpath.join(source_dir, target))
                        extension = Path(archive_member).suffix.lower()
                        if extension not in BROWSER_IMAGE_EXTENSIONS:
                            continue
                        try:
                            blob = archive.read(archive_member)
                        except KeyError:
                            continue
                        if len(blob) < MIN_SLIDE_IMAGE_BYTES:
                            continue
                        candidates.append((len(blob), Path(archive_member).name, blob))

                    if not candidates:
                        continue

                    candidates.sort(key=lambda item: (-item[0], item[1]))
                    size_bytes, image_name, blob = candidates[0]
                    deck_slug = slugify(str(Path(doc.relative_path).with_suffix("")))
                    extension = Path(image_name).suffix.lower()
                    output_name = f"slide-{slide_index:03d}-{short_hash(f'{doc.relative_path}|{slide_index}|{image_name}|{size_bytes}')}{extension}"
                    relative_image_path = Path("images") / deck_slug / output_name
                    absolute_image_path = artifact_dir / relative_image_path
                    ensure_dir(absolute_image_path.parent)
                    absolute_image_path.write_bytes(blob)

                    location_label = "page" if str(doc.source_type).lower() == "pdf" else "slide"
                    records.append(
                        SlideImageRecord(
                            doc_id=doc.doc_id,
                            relative_path=doc.relative_path,
                            slide_index=int(doc.unit_index or slide_index),
                            image_path=relative_image_path.as_posix(),
                            image_caption=f"Lecture figure from {Path(doc.relative_path).as_posix()} {location_label} {int(doc.unit_index or slide_index)}",
                            image_name=image_name,
                            size_bytes=size_bytes,
                        )
                    )
        except Exception as exc:
            issues.append({"relative_path": image_source_relative_path, "issue": f"{type(exc).__name__}: {exc}"})

    return records, issues


def build_inventory(
    course_root: Path,
    course_files: Sequence[Path],
    raw_documents: Sequence[RawDocument],
    issues: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    format_counts = Counter(path.suffix.lower() for path in course_files)
    week_file_counts = Counter(detect_week(path.relative_to(course_root)) for path in course_files)
    week_unit_counts = Counter(doc.week for doc in raw_documents)
    titles_by_week: Dict[str, List[str]] = defaultdict(list)
    for doc in raw_documents:
        if len(titles_by_week[doc.week]) < 8:
            titles_by_week[doc.week].append(doc.title)

    return {
        "course_root": str(course_root),
        "file_count": len(course_files),
        "raw_document_count": len(raw_documents),
        "format_counts": dict(sorted(format_counts.items())),
        "week_file_counts": dict(sorted(week_file_counts.items())),
        "week_unit_counts": dict(sorted(week_unit_counts.items())),
        "sample_titles_by_week": {key: value for key, value in sorted(titles_by_week.items())},
        "issues": list(issues),
    }


def inventory_markdown(inventory: Dict[str, Any]) -> str:
    lines = [
        "# Course 4186 Inventory",
        "",
        f"- Course root: `{inventory['course_root']}`",
        f"- File count: `{inventory['file_count']}`",
        f"- Extracted units: `{inventory['raw_document_count']}`",
        "",
        "## File formats",
        "",
    ]
    for suffix, count in inventory["format_counts"].items():
        lines.append(f"- `{suffix}`: {count}")
    lines.extend(["", "## Files by week", ""])
    for week, count in inventory["week_file_counts"].items():
        lines.append(f"- `{week}`: {count}")
    lines.extend(["", "## Extracted units by week", ""])
    for week, count in inventory["week_unit_counts"].items():
        lines.append(f"- `{week}`: {count}")
    lines.extend(["", "## Sample titles by week", ""])
    for week, titles in inventory["sample_titles_by_week"].items():
        joined = "; ".join(titles[:5])
        lines.append(f"- `{week}`: {joined}")
    if inventory["issues"]:
        lines.extend(["", "## Extraction issues", ""])
        for issue in inventory["issues"]:
            lines.append(f"- `{issue['relative_path']}`: {issue['issue']}")
    return "\n".join(lines) + "\n"


def analyze_only(course_root: Path, artifact_dir: Path, max_week: Optional[int] = None) -> Dict[str, Any]:
    ensure_dir(artifact_dir)
    raw_documents, issues, course_files = extract_raw_documents(course_root, max_week=max_week)
    inventory = build_inventory(course_root, course_files, raw_documents, issues)

    write_json(artifact_dir / "inventory.json", inventory)
    (artifact_dir / "inventory.md").write_text(inventory_markdown(inventory), encoding="utf-8")
    write_jsonl(artifact_dir / "raw_documents.jsonl", (asdict(doc) for doc in raw_documents))

    return {
        "inventory": inventory,
        "raw_document_count": len(raw_documents),
        "issue_count": len(issues),
    }


COURSE_TAXONOMY: List[Dict[str, Any]] = [
    {
        "name": "Computer vision foundations",
        "start_week": 1,
        "description": "Understanding what computer vision studies: how images are represented, interpreted, and used for recognition, measurement, and scene understanding.",
        "keywords": [
            "computer vision",
            "goal of computer vision",
            "what is computer vision",
            "why is computer vision difficult",
            "viewpoint variation in vision",
            "background clutter in vision",
            "occlusion in vision",
            "high-level understanding from images",
        ],
    },
    {
        "name": "Image filtering and convolution",
        "start_week": 1,
        "description": "Representing images as grids and applying linear filters, convolution, correlation, smoothing, and sharpening.",
        "keywords": [
            "image filtering",
            "linear filter",
            "convolution",
            "cross-correlation",
            "gaussian filter",
            "mean filter",
            "image sharpening",
        ],
    },
    {
        "name": "Edge detection",
        "start_week": 2,
        "description": "Detecting visual discontinuities to capture salient boundaries and structure in images.",
        "keywords": ["edge detection", "edges", "discontinuities", "visual changes"],
    },
    {
        "name": "Image resampling",
        "start_week": 2,
        "description": "Changing image resolution or coordinate sampling while preserving useful information.",
        "keywords": [
            "image resampling",
            "image scaling",
            "sub-sampling",
            "upsampling",
            "interpolation",
            "nearest neighbor interpolation",
            "aliasing",
        ],
    },
    {
        "name": "Texture analysis",
        "start_week": 3,
        "description": "Describing repeated local appearance patterns and using them for recognition or classification.",
        "keywords": ["texture", "natural textures", "color texture"],
    },
    {
        "name": "Convolutional neural networks",
        "start_week": 4,
        "description": "Learning feature hierarchies for image classification and recognition using CNN architectures.",
        "keywords": ["convolutional neural network", "cnn", "classification", "decision boundary"],
    },
    {
        "name": "Image segmentation",
        "start_week": 4,
        "description": "Partitioning an image into meaningful regions or objects based on appearance and context.",
        "keywords": ["image segmentation", "segmentation", "regions", "zebras"],
    },
    {
        "name": "Harris corners and local features",
        "start_week": 5,
        "description": "Detecting repeatable corner-like structures and describing local image evidence around them.",
        "keywords": ["harris corner", "corner detection", "feature extraction", "local features"],
    },
    {
        "name": "SIFT and scale-invariant features",
        "start_week": 5,
        "description": "Building descriptors that remain stable across scale and orientation changes.",
        "keywords": ["scale invariant feature transform", "sift", "difference of gaussian", "descriptor"],
    },
    {
        "name": "Bag-of-words image retrieval",
        "start_week": 6,
        "description": "Representing images as visual word histograms to support matching and large-scale retrieval.",
        "keywords": ["bag of words", "bag-of-words", "image matching", "visual words"],
    },
    {
        "name": "Geometric transformations",
        "start_week": 6,
        "description": "Modeling translation, rotation, scaling, and other geometric relationships between images.",
        "keywords": ["transformations", "similarity transformation", "translation", "rotation", "uniform scale"],
    },
    {
        "name": "Image alignment",
        "start_week": 6,
        "description": "Computing transformations that align multiple images using feature matches and consistency criteria.",
        "keywords": ["image alignment", "align", "compute the transform", "matches between images"],
    },
    {
        "name": "Camera model and projection",
        "start_week": 8,
        "description": "Understanding pinhole projection, focal length, field of view, and how 3D scenes map to images.",
        "keywords": ["camera model", "pinhole camera", "focal length", "field of view", "vanishing point", "cameras"],
    },
    {
        "name": "Stereo vision and depth",
        "start_week": 10,
        "description": "Recovering depth or 3D structure from multiple views by finding correspondences across images.",
        "keywords": ["stereo vision", "depth map", "stereo reconstruction", "correspondence", "disparity", "rectification"],
    },
    {
        "name": "Epipolar geometry",
        "start_week": 10,
        "description": "Constraining stereo correspondences with epipolar lines and geometric relations.",
        "keywords": ["epipolar", "epipolar line", "epipolar plane", "epipolar constraint", "essential matrix", "fundamental matrix"],
    },
    {
        "name": "Structure from motion",
        "start_week": 10,
        "description": "Estimating scene structure and camera motion from image sequences or multiple views.",
        "keywords": ["structure from motion", "point cloud", "camera motion"],
    },
    {
        "name": "Motion and optical flow",
        "start_week": 11,
        "description": "Estimating pixel or feature motion across frames to understand dynamic scenes.",
        "keywords": ["motion", "optical flow", "brightness constancy", "Lucas-Kanade", "aperture problem"],
    },
    {
        "name": "LLM prompting and DeepSeek overview",
        "start_week": 7,
        "description": "A special-topic lecture on large language models, prompting methods, and DeepSeek evolution.",
        "keywords": ["deepseek", "llm", "large language model", "chain of thought", "prompting"],
    },
]


def split_text_fallback(text: str, chunk_size: int, overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    pieces: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        window = text[start:end]
        if end < len(text):
            cut = window.rfind(" ")
            if cut > int(chunk_size * 0.6):
                end = start + cut
                window = text[start:end]
        pieces.append(window.strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return [piece for piece in pieces if piece]


def build_chunk_records(
    raw_documents: Sequence[RawDocument],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[ChunkRecord]:
    records: List[ChunkRecord] = []

    if Document is not None and RecursiveCharacterTextSplitter is not None:
        source_docs = [
            Document(
                page_content=doc.text,
                metadata={
                    "doc_id": doc.doc_id,
                    "relative_path": doc.relative_path,
                    "source_type": doc.source_type,
                    "week": doc.week,
                    "title": doc.title,
                    "unit_type": doc.unit_type,
                    "unit_index": doc.unit_index,
                },
            )
            for doc in raw_documents
        ]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " "],
        )
        split_docs = splitter.split_documents(source_docs)
        chunk_counter: Counter[str] = Counter()
        for split_doc in split_docs:
            doc_id = split_doc.metadata["doc_id"]
            chunk_counter[doc_id] += 1
            records.append(
                ChunkRecord(
                    chunk_id=f"chunk-{short_hash(f'{doc_id}|{chunk_counter[doc_id]}')}",
                    doc_id=doc_id,
                    relative_path=split_doc.metadata["relative_path"],
                    source_type=split_doc.metadata["source_type"],
                    week=split_doc.metadata["week"],
                    title=split_doc.metadata["title"],
                    unit_type=split_doc.metadata["unit_type"],
                    unit_index=int(split_doc.metadata["unit_index"]),
                    chunk_index=chunk_counter[doc_id],
                    text=normalize_text(split_doc.page_content),
                )
            )
        return [record for record in records if record.text]

    for doc in raw_documents:
        for index, piece in enumerate(split_text_fallback(doc.text, chunk_size, overlap), start=1):
            records.append(
                ChunkRecord(
                    chunk_id=f"chunk-{short_hash(f'{doc.doc_id}|{index}')}",
                    doc_id=doc.doc_id,
                    relative_path=doc.relative_path,
                    source_type=doc.source_type,
                    week=doc.week,
                    title=doc.title,
                    unit_type=doc.unit_type,
                    unit_index=doc.unit_index,
                    chunk_index=index,
                    text=normalize_text(piece),
                )
            )
    return [record for record in records if record.text]


def dense_stack_available() -> bool:
    return all(item is not None for item in [Document, HuggingFaceEmbeddings, FAISS])


def get_embeddings(model_name: str) -> Any:
    if not dense_stack_available():
        raise RuntimeError("Dense retrieval dependencies are not available in the current Python environment.")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


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


def build_knowledge_points(chunk_records: Sequence[ChunkRecord]) -> List[KnowledgePoint]:
    knowledge_points: List[KnowledgePoint] = []
    for entry in COURSE_TAXONOMY:
        start_week = int(entry.get("start_week", 1))
        max_supported_week = max((parse_week_number(chunk.week) or 0) for chunk in chunk_records) if chunk_records else 0
        if max_supported_week and start_week > max_supported_week:
            continue
        scored: List[Tuple[float, ChunkRecord, List[str]]] = []
        best_by_keyword: Dict[str, Tuple[float, ChunkRecord]] = {}
        for chunk in chunk_records:
            blob = "\n".join([chunk.title, chunk.relative_path, chunk.text]).lower()
            title_blob = "\n".join([chunk.title, chunk.relative_path]).lower()
            matched_keywords: List[str] = []
            score = 0.0
            for keyword in entry["keywords"]:
                term = keyword.lower()
                if term in blob:
                    matched_keywords.append(keyword)
                    score += 3.0 if term in title_blob else 1.5
            if not matched_keywords:
                continue
            if len(matched_keywords) > 1:
                score += min(len(matched_keywords) - 1, 3) * 0.9

            relative_path = normalized_relative_path(chunk.relative_path)
            week_number = parse_week_number(chunk.week) or start_week
            if not is_revision_path(relative_path):
                score -= abs(week_number - start_week) * 0.6
            else:
                score -= 1.5
            if "lecture" in relative_path:
                score += 2.0
            if str(chunk.source_type).lower() in {"pptx", "pdf"}:
                score += 1.0
            if "tutorial/" in relative_path:
                score -= 1.2
            if "question" in relative_path:
                score -= 3.5
            if "answer" in relative_path:
                score -= 2.5
            if score > 0:
                scored.append((score, chunk, matched_keywords))
                for keyword in matched_keywords:
                    current = best_by_keyword.get(keyword)
                    if current is None or support_sort_key(score, chunk) < support_sort_key(current[0], current[1]):
                        best_by_keyword[keyword] = (score, chunk)
        if not scored:
            continue

        scored.sort(key=lambda pair: support_sort_key(pair[0], pair[1]))
        selected: List[ChunkRecord] = []
        seen_chunk_ids: set[str] = set()
        seen_units: set[Tuple[str, str, int]] = set()

        def try_add(chunk: ChunkRecord) -> bool:
            unit_key = source_unit_family_key(chunk.relative_path, chunk.unit_type, chunk.unit_index)
            if chunk.chunk_id in seen_chunk_ids or unit_key in seen_units:
                return False
            selected.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
            seen_units.add(unit_key)
            return True

        for keyword in entry["keywords"]:
            candidate = best_by_keyword.get(keyword)
            if candidate is None:
                continue
            chunk = candidate[1]
            try_add(chunk)
            if len(selected) >= 10:
                break

        for score, chunk, _matched_keywords in scored:
            if len(selected) >= 10:
                break
            if is_low_priority_review_source(str(chunk.relative_path), str(chunk.source_type)):
                continue
            try_add(chunk)

        for score, chunk, _matched_keywords in scored:
            if len(selected) >= 10:
                break
            try_add(chunk)

        knowledge_points.append(
            KnowledgePoint(
                kp_id=f"kp-{slugify(entry['name'])}",
                name=entry["name"],
                description=entry["description"],
                keywords=list(entry["keywords"]),
                weeks=sorted({chunk.week for chunk in selected}),
                source_files=sorted({chunk.relative_path for chunk in selected}),
                support_chunk_ids=[chunk.chunk_id for chunk in selected],
                support_preview=[safe_snippet(chunk.text, limit=200) for chunk in selected[:3]],
            )
        )
    return knowledge_points


def build_dense_index(documents: List[Any], index_dir: Path, model_name: str) -> Dict[str, Any]:
    try:
        embeddings = get_embeddings(model_name)
        vector_store = FAISS.from_documents(documents, embeddings)
        ensure_dir(index_dir)
        vector_store.save_local(str(index_dir))
        return {
            "status": "built",
            "index_dir": str(index_dir),
            "count": len(documents),
            "model_name": model_name,
        }
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"{type(exc).__name__}: {exc}",
            "index_dir": str(index_dir),
            "count": len(documents),
            "model_name": model_name,
        }


def build_chunk_dense_index(
    chunk_records: Sequence[ChunkRecord],
    index_dir: Path,
    model_name: str,
) -> Dict[str, Any]:
    if Document is None:
        return {
            "status": "skipped",
            "reason": "langchain_core is unavailable",
            "index_dir": str(index_dir),
            "count": len(chunk_records),
            "model_name": model_name,
        }
    documents = [
        Document(
            page_content=record.text,
            metadata={
                "chunk_id": record.chunk_id,
                "relative_path": record.relative_path,
                "week": record.week,
                "title": record.title,
                "source_type": record.source_type,
                "unit_type": record.unit_type,
                "unit_index": record.unit_index,
            },
        )
        for record in chunk_records
    ]
    return build_dense_index(documents, index_dir, model_name)


def build_kp_dense_index(
    knowledge_points: Sequence[KnowledgePoint],
    index_dir: Path,
    model_name: str,
) -> Dict[str, Any]:
    if Document is None:
        return {
            "status": "skipped",
            "reason": "langchain_core is unavailable",
            "index_dir": str(index_dir),
            "count": len(knowledge_points),
            "model_name": model_name,
        }
    documents = [
        Document(
            page_content="\n".join(
                [
                    kp.name,
                    kp.description,
                    "Keywords: " + ", ".join(kp.keywords),
                    "Sources: " + ", ".join(kp.source_files[:5]),
                ]
            ),
            metadata={
                "kp_id": kp.kp_id,
                "name": kp.name,
            },
        )
        for kp in knowledge_points
    ]
    return build_dense_index(documents, index_dir, model_name)


def split_sentences(text: str) -> List[str]:
    compact = WS_RE.sub(" ", text).strip()
    if not compact:
        return []
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", compact) if sentence.strip()]


def extractive_summary(texts: Sequence[str], fallback: str) -> str:
    picked: List[str] = []
    for text in texts:
        for sentence in split_sentences(text):
            if 35 <= len(sentence) <= 240 and sentence not in picked:
                picked.append(sentence)
            if len(picked) >= 2:
                return " ".join(picked)
    return fallback


def choose_distractors(correct_name: str, all_names: Sequence[str], seed: str, count: int = 3) -> List[str]:
    candidates = [name for name in all_names if name != correct_name]
    if len(candidates) <= count:
        return candidates
    rng = random.Random(seed)
    return rng.sample(candidates, count)


def dedupe_texts(values: Sequence[str]) -> List[str]:
    deduped: List[str] = []
    seen: set[str] = set()
    for value in values:
        clean = WS_RE.sub(" ", str(value or "")).strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(clean)
    return deduped


def keyword_bundle(kp: KnowledgePoint, size: int = 3) -> str:
    values = [clean_display_text(item) for item in dedupe_texts([kp.name] + list(kp.keywords))]
    return ", ".join(values[:size])


def primary_keyword(kp: KnowledgePoint) -> str:
    values = [clean_display_text(item) for item in dedupe_texts(list(kp.keywords) + [kp.name])]
    return values[0] if values else kp.name


def scenario_for_kp(kp: KnowledgePoint) -> str:
    desc = clean_display_text(kp.description[:1].lower() + kp.description[1:] if kp.description else kp.name)
    return safe_snippet(f"A computer vision task requires {desc}", 150)


def question_study_note(kp: KnowledgePoint) -> str:
    note = f"{clean_display_text(kp.description)} Key cues: {keyword_bundle(kp)}."
    return safe_snippet(note, 220)


def build_option_lines(correct_text: str, distractors: Sequence[str], seed: str) -> Tuple[List[str], str]:
    correct_value = clean_display_text(correct_text)
    option_pool = dedupe_texts([correct_value] + [clean_display_text(item) for item in distractors])
    if len(option_pool) < 2:
        option_pool = dedupe_texts([correct_value, "None of the above"])
    rng = random.Random(seed)
    rng.shuffle(option_pool)
    labels = ["A", "B", "C", "D"][: len(option_pool)]
    correct_option = labels[option_pool.index(correct_value)]
    options = [f"{label}. {text}" for label, text in zip(labels, option_pool)]
    return options, correct_option


def section_label_for_chunk(chunk: ChunkRecord) -> str:
    title = clean_display_text(chunk.title)
    stem = clean_display_text(PurePosixPath(str(chunk.relative_path).replace("\\", "/")).stem)
    title_tokens = set(tokenize(title))
    stem_tokens = set(tokenize(stem))
    looks_generic = (
        not title
        or title.lower() == stem.lower()
        or (title_tokens and title_tokens.issubset(stem_tokens) and len(title_tokens) <= 4)
        or len(title) <= 6
    )
    if looks_generic:
        fallback = safe_snippet(clean_display_text(chunk.text), 110)
        return fallback or title or stem
    return title


def normalized_relative_path(relative_path: str) -> str:
    return str(relative_path or "").replace("\\", "/").lower()


def is_revision_path(relative_path: str) -> bool:
    return "revision" in normalized_relative_path(relative_path)


def is_low_priority_review_source(relative_path: str, source_type: str) -> bool:
    path = normalized_relative_path(relative_path)
    return (
        is_revision_path(path)
        or "question" in path
        or "answer" in path
        or "tutorial/" in path
        or str(source_type or "").lower() == "docx"
    )


def support_sort_key(score: float, chunk: ChunkRecord) -> Tuple[float, int, int, str, int]:
    return (
        -float(score),
        1 if is_revision_path(str(chunk.relative_path)) else 0,
        SOURCE_TYPE_PRIORITY.get(str(chunk.source_type).lower(), 3),
        str(chunk.relative_path),
        int(chunk.unit_index or 0),
    )


def review_path_penalty(relative_path: str, source_type: str) -> float:
    path = normalized_relative_path(relative_path)
    penalty = 0.0
    if "revision" in path:
        penalty += 2.0
    if "question" in path:
        penalty += 4.5
    if "answer" in path:
        penalty += 3.0
    if "tutorial/" in path:
        penalty += 1.2
    if str(source_type or "").lower() == "docx":
        penalty += 0.8
    return penalty


def review_token_set(*texts: str) -> set[str]:
    tokens: set[str] = set()
    for text in texts:
        for token in tokenize(text):
            if len(token) <= 2 or token in REVIEW_STOPWORDS:
                continue
            tokens.add(token)
    return tokens


def rank_review_chunks(
    prompt: str,
    correct_text: str,
    explanation: str,
    support_chunks: Sequence[ChunkRecord],
    review_terms: Optional[Sequence[str]] = None,
) -> List[Tuple[float, ChunkRecord]]:
    if not support_chunks:
        return []

    query = " ".join(
        part for part in [
            clean_display_text(prompt),
            clean_display_text(correct_text),
            clean_display_text(explanation),
            " ".join(clean_display_text(term) for term in (review_terms or [])),
        ]
        if part
    ).strip()
    query_terms = review_token_set(query)
    anchor_terms = review_token_set(" ".join(clean_display_text(term) for term in (review_terms or [])))

    lexical_hits = lexical_rank(
        query or clean_display_text(correct_text),
        support_chunks,
        text_getter=lambda chunk: "\n".join([chunk.title, chunk.relative_path, chunk.text]),
        top_k=len(support_chunks),
    )
    lexical_scores: Dict[str, float] = {
        chunk.chunk_id: float(score)
        for score, chunk in lexical_hits
    }

    ranked: List[Tuple[float, ChunkRecord]] = []
    seen: set[Tuple[str, str, int]] = set()
    for chunk in support_chunks:
        key = (chunk.relative_path, chunk.unit_type, chunk.unit_index)
        if key in seen:
            continue
        seen.add(key)

        title_text = clean_display_text(chunk.title)
        section_text = section_label_for_chunk(chunk)
        path_text = str(chunk.relative_path).replace("\\", "/")
        title_terms = review_token_set(title_text)
        section_terms = review_token_set(section_text)
        path_terms = review_token_set(path_text)
        body_terms = review_token_set(chunk.text[:600])
        combined_terms = title_terms | section_terms | path_terms | body_terms

        title_overlap = len(query_terms & title_terms)
        section_overlap = len(query_terms & section_terms)
        path_overlap = len(query_terms & path_terms)
        body_overlap = len(query_terms & body_terms)
        anchor_overlap = len(anchor_terms & combined_terms) if anchor_terms else 0

        exact_phrase_bonus = 0.0
        for term in review_terms or []:
            phrase = clean_display_text(str(term or "")).lower()
            if not phrase:
                continue
            haystack = " ".join([title_text, section_text, chunk.text[:600]]).lower()
            if phrase in haystack:
                exact_phrase_bonus += 3.0

        score = (
            lexical_scores.get(chunk.chunk_id, 0.0)
            + title_overlap * 4.5
            + section_overlap * 3.5
            + path_overlap * 3.0
            + body_overlap * 1.5
            + anchor_overlap * 5.0
            + exact_phrase_bonus
            + (2.0 - float(SOURCE_TYPE_PRIORITY.get(str(chunk.source_type).lower(), 3)))
        )

        relative_path = path_text.lower()
        if "revision" in relative_path and anchor_overlap == 0 and (title_overlap + section_overlap) == 0:
            score -= 3.5
        score -= review_path_penalty(relative_path, str(chunk.source_type))
        if not (title_overlap or section_overlap or path_overlap or body_overlap or anchor_overlap or lexical_scores.get(chunk.chunk_id)):
            score -= 5.0

        ranked.append((score, chunk))

    ranked.sort(
        key=lambda item: (
            -item[0],
            SOURCE_TYPE_PRIORITY.get(str(item[1].source_type).lower(), 3),
            item[1].relative_path,
            item[1].unit_index,
        )
    )
    return ranked


def build_review_refs(
    prompt: str,
    correct_text: str,
    explanation: str,
    support_chunks: Sequence[ChunkRecord],
    review_terms: Optional[Sequence[str]] = None,
    preferred_chunks: Optional[Sequence[ChunkRecord]] = None,
    max_refs: int = 2,
) -> List[Dict[str, Any]]:
    def ref_key(ref: Dict[str, Any]) -> Tuple[str, str, int]:
        try:
            unit_index = int(ref.get("unit_index", 0) or 0)
        except Exception:
            unit_index = 0
        display_source = canonical_display_source_name(str(ref.get("display_source") or ref.get("source") or ""))
        resolved_unit_type = display_unit_type(display_source, str(ref.get("unit_type") or ""))
        return (
            normalized_relative_path(display_source),
            resolved_unit_type,
            unit_index,
        )

    def chunk_to_ref(chunk: ChunkRecord) -> Dict[str, Any]:
        source_path = str(chunk.relative_path).replace("\\", "/")
        display_source = canonical_display_source_name(source_path)
        resolved_unit_type = display_unit_type(display_source, chunk.unit_type)
        return {
            "source": str(chunk.relative_path),
            "display_source": display_source,
            "location": display_source_location(
                relative_path=source_path,
                source_type=chunk.source_type,
                unit_type=chunk.unit_type,
                unit_index=chunk.unit_index,
                display_source=display_source,
            ),
            "section": section_label_for_chunk(chunk),
            "unit_type": resolved_unit_type,
            "unit_index": chunk.unit_index,
        }

    ranked = rank_review_chunks(
        prompt=prompt,
        correct_text=correct_text,
        explanation=explanation,
        support_chunks=support_chunks,
        review_terms=review_terms,
    )
    if not ranked:
        return []

    top_score = ranked[0][0]
    strong_threshold = max(2.0, top_score * 0.55)
    relaxed_threshold = max(1.5, top_score * 0.35)
    viable_ranked = [(score, chunk) for score, chunk in ranked if score >= strong_threshold]
    if not viable_ranked:
        viable_ranked = ranked[:1]

    preferred_ranked = [
        (score, chunk)
        for score, chunk in viable_ranked
        if not is_low_priority_review_source(str(chunk.relative_path), str(chunk.source_type))
    ]
    if not preferred_ranked:
        preferred_ranked = [
            (score, chunk)
            for score, chunk in ranked
            if score >= relaxed_threshold and not is_low_priority_review_source(str(chunk.relative_path), str(chunk.source_type))
        ]
    if not preferred_ranked:
        preferred_ranked = viable_ranked

    refs: List[Dict[str, Any]] = []
    seen_ref_keys: set[Tuple[str, str, int]] = set()

    prioritized_chunks = sorted(
        list(preferred_chunks or []),
        key=lambda chunk: (
            0 if str(chunk.source_type).lower() == "pdf" else 1,
            SOURCE_TYPE_PRIORITY.get(str(chunk.source_type).lower(), 3),
            str(chunk.relative_path),
            int(chunk.unit_index or 0),
            int(chunk.chunk_index or 0),
        ),
    )
    for chunk in prioritized_chunks:
        ref = chunk_to_ref(chunk)
        key = ref_key(ref)
        if key in seen_ref_keys:
            continue
        refs.append(ref)
        seen_ref_keys.add(key)
        if len(refs) >= max_refs:
            return refs

    for score, chunk in preferred_ranked:
        ref = chunk_to_ref(chunk)
        key = ref_key(ref)
        if key in seen_ref_keys:
            continue
        refs.append(ref)
        seen_ref_keys.add(key)
        if len(refs) >= max_refs:
            break
    return refs


def review_chunks_for_image(
    image_record: SlideImageRecord,
    support_chunks: Sequence[ChunkRecord],
) -> List[ChunkRecord]:
    image_family = source_family_key(str(image_record.relative_path))
    try:
        slide_index = int(image_record.slide_index)
    except (TypeError, ValueError):
        return []

    matches = [
        chunk
        for chunk in support_chunks
        if source_family_key(str(chunk.relative_path)) == image_family
        and int(chunk.unit_index or 0) == slide_index
    ]
    matches.sort(
        key=lambda chunk: (
            0 if str(chunk.source_type).lower() == "pdf" else 1,
            SOURCE_TYPE_PRIORITY.get(str(chunk.source_type).lower(), 3),
            str(chunk.relative_path),
            int(chunk.chunk_index or 0),
        )
    )
    return matches


def review_chunks_for_reference(
    source_path: Optional[str],
    unit_index: Any,
    support_chunks: Sequence[ChunkRecord],
) -> List[ChunkRecord]:
    if not source_path:
        return []
    try:
        normalized_index = int(unit_index or 0)
    except (TypeError, ValueError):
        return []
    target_family = source_family_key(str(source_path))
    matches = [
        chunk
        for chunk in support_chunks
        if source_family_key(str(chunk.relative_path)) == target_family
        and int(chunk.unit_index or 0) == normalized_index
    ]
    matches.sort(
        key=lambda chunk: (
            0 if str(chunk.source_type).lower() == "pdf" else 1,
            SOURCE_TYPE_PRIORITY.get(str(chunk.source_type).lower(), 3),
            str(chunk.relative_path),
            int(chunk.chunk_index or 0),
        )
    )
    return matches


def image_caption_for_question(
    image_record: SlideImageRecord,
    review_refs: Sequence[Dict[str, Any]],
) -> str:
    if review_refs:
        primary = review_refs[0]
        return clean_display_text(
            f"Lecture figure from {primary.get('display_source') or canonical_display_source_name(image_record.relative_path)} {primary.get('location') or f'page {image_record.slide_index}'}"
        )
    return clean_display_text(
        f"Lecture figure from {canonical_display_source_name(image_record.relative_path)} page {image_record.slide_index}"
    )


def build_mcq_question(
    question_id: str,
    kp: KnowledgePoint,
    prompt: str,
    correct_text: str,
    distractors: Sequence[str],
    explanation: str,
    review_chunks: Sequence[ChunkRecord],
    source_chunk_ids: Sequence[str],
    source_files: Sequence[str],
    image_path: Optional[str] = None,
    image_caption: Optional[str] = None,
    review_terms: Optional[Sequence[str]] = None,
    review_refs_override: Optional[Sequence[Dict[str, Any]]] = None,
) -> QuestionRecord:
    options, correct_option = build_option_lines(correct_text, distractors, question_id)
    review_refs = [
        dict(item)
        for item in (
            review_refs_override
            if review_refs_override is not None
            else build_review_refs(
                prompt,
                correct_text,
                explanation,
                review_chunks,
                review_terms=review_terms,
            )
        )
    ]
    return QuestionRecord(
        question_id=question_id,
        kp_id=kp.kp_id,
        kp_name=kp.name,
        question_type="multiple_choice",
        question=clean_display_text(prompt),
        answer=clean_display_text(correct_text),
        explanation=clean_display_text(explanation),
        options=options,
        correct_option=correct_option,
        source_chunk_ids=list(source_chunk_ids),
        source_files=list(source_files),
        review_refs=review_refs,
        image_path=image_path,
        image_caption=clean_display_text(image_caption) if image_caption else None,
    )


def select_question_image(
    support_chunks: Sequence[ChunkRecord],
    slide_images_by_doc_id: Dict[str, List[SlideImageRecord]],
    prompt: str,
    correct_text: str,
    explanation: str,
    review_terms: Optional[Sequence[str]] = None,
) -> Optional[SlideImageRecord]:
    ranked_chunks = rank_review_chunks(
        prompt=prompt,
        correct_text=correct_text,
        explanation=explanation,
        support_chunks=support_chunks,
        review_terms=review_terms,
    )
    if not ranked_chunks:
        return None

    preferred_ranked = [
        (score, chunk)
        for score, chunk in ranked_chunks
        if not is_low_priority_review_source(str(chunk.relative_path), str(chunk.source_type))
    ] or ranked_chunks
    top_score, top_chunk = preferred_ranked[0]

    if str(top_chunk.source_type).lower() == "pdf" and str(top_chunk.unit_type).lower() == "page":
        candidates = [
            item for item in slide_images_by_doc_id.get(top_chunk.doc_id, [])
            if int(item.slide_index or 0) == int(top_chunk.unit_index or 0)
        ]
        if candidates:
            candidates.sort(key=lambda item: (-item.size_bytes, item.relative_path, item.slide_index))
            return candidates[0]

    if str(top_chunk.source_type).lower() in {"pptx", "ppt"} and str(top_chunk.unit_type).lower() == "slide":
        candidates = list(slide_images_by_doc_id.get(top_chunk.doc_id, []))
        if candidates:
            candidates.sort(key=lambda item: (-item.size_bytes, item.relative_path, item.slide_index))
            return candidates[0]
        return None

    top_terms = review_token_set(top_chunk.title, top_chunk.text[:260])
    for score, chunk in preferred_ranked[1:]:
        if score < max(2.5, top_score * 0.72):
            break
        if str(chunk.source_type).lower() not in {"pptx", "ppt"} or str(chunk.unit_type).lower() != "slide":
            continue
        if chunk.week != top_chunk.week or int(chunk.unit_index or 0) != int(top_chunk.unit_index or 0):
            continue
        candidate_terms = review_token_set(chunk.title, chunk.text[:260])
        if top_terms and len(top_terms & candidate_terms) < min(2, len(top_terms)):
            continue
        candidates = list(slide_images_by_doc_id.get(chunk.doc_id, []))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (-item.size_bytes, item.relative_path, item.slide_index))
        return candidates[0]
    return None


def select_overridden_question_image(
    slide_images_by_source: Dict[Tuple[str, int], SlideImageRecord],
    source_path: Optional[str],
    slide_index: Any,
) -> Optional[SlideImageRecord]:
    if not source_path:
        return None
    try:
        normalized_slide_index = int(slide_index)
    except (TypeError, ValueError):
        return None
    normalized_source = normalized_relative_path(str(source_path))
    direct = slide_images_by_source.get((normalized_source, normalized_slide_index))
    if direct is not None:
        return direct

    source_posix = clean_display_text(str(source_path)).replace("\\", "/")
    source_pure = PurePosixPath(source_posix)
    if source_pure.suffix.lower() in {".ppt", ".pptx", ".doc", ".docx"}:
        pdf_source = source_pure.with_suffix(".pdf").as_posix().lower()
        direct_pdf = slide_images_by_source.get((pdf_source, normalized_slide_index))
        if direct_pdf is not None:
            return direct_pdf

    source_parent = str(source_pure.parent).lower()
    source_key = material_stem_key(source_pure.stem)
    for (candidate_path, candidate_index), image in slide_images_by_source.items():
        if int(candidate_index) != normalized_slide_index:
            continue
        candidate_pure = PurePosixPath(candidate_path)
        if str(candidate_pure.parent).lower() != source_parent:
            continue
        if material_stem_key(candidate_pure.stem) == source_key:
            return image
    return None


def fallback_questions_for_kp(
    kp: KnowledgePoint,
    support_chunks: Sequence[ChunkRecord],
    review_chunks: Sequence[ChunkRecord],
    all_chunk_records: Sequence[ChunkRecord],
    all_kps: Sequence[KnowledgePoint],
    slide_images_by_doc_id: Dict[str, List[SlideImageRecord]],
    slide_images_by_source: Dict[Tuple[str, int], SlideImageRecord],
    question_count: int,
) -> List[QuestionRecord]:
    question_count = max(MIN_QUESTION_COUNT, question_count)
    source_files = sorted({chunk.relative_path for chunk in support_chunks})
    source_chunk_ids = [chunk.chunk_id for chunk in support_chunks]
    specs = QUESTION_BLUEPRINTS.get(kp.name, [])
    questions: List[QuestionRecord] = []
    for index, spec in enumerate(specs[:question_count], start=1):
        selected_image = select_overridden_question_image(
            slide_images_by_source,
            source_path=spec.get("image_source"),
            slide_index=spec.get("image_slide"),
        )
        if selected_image is None:
            selected_image = select_question_image(
                review_chunks,
                slide_images_by_doc_id,
                prompt=str(spec.get("prompt", "")).strip(),
                correct_text=str(spec.get("correct", "")).strip(),
                explanation=str(spec.get("explanation", "")).strip(),
                review_terms=spec.get("review_terms"),
            )
        use_image = bool(spec.get("use_image")) and selected_image is not None
        explicit_review_chunks = review_chunks_for_reference(
            spec.get("review_source"),
            spec.get("review_page"),
            all_chunk_records,
        )
        explicit_review_refs: Optional[List[Dict[str, Any]]] = None
        if explicit_review_chunks:
            explicit_review_refs = build_review_refs(
                str(spec.get("prompt", "")).strip(),
                str(spec.get("correct", "")).strip(),
                str(spec.get("explanation", "")).strip(),
                explicit_review_chunks,
                review_terms=spec.get("review_terms"),
                preferred_chunks=explicit_review_chunks,
            )
        image_review_refs: Optional[List[Dict[str, Any]]] = None
        image_caption: Optional[str] = None
        if use_image and selected_image is not None:
            anchored_chunks = review_chunks_for_image(selected_image, review_chunks)
            image_review_refs = build_review_refs(
                str(spec.get("prompt", "")).strip(),
                str(spec.get("correct", "")).strip(),
                str(spec.get("explanation", "")).strip(),
                review_chunks,
                review_terms=spec.get("review_terms"),
                preferred_chunks=anchored_chunks,
            )
            image_caption = image_caption_for_question(selected_image, image_review_refs)
        review_refs_override = explicit_review_refs if explicit_review_refs is not None else image_review_refs
        questions.append(
            build_mcq_question(
                question_id=f"{kp.kp_id}-q{index}",
                kp=kp,
                prompt=str(spec.get("prompt", "")).strip(),
                correct_text=str(spec.get("correct", "")).strip(),
                distractors=[str(item).strip() for item in spec.get("distractors", [])],
                explanation=str(spec.get("explanation", "")).strip(),
                review_chunks=review_chunks,
                source_chunk_ids=source_chunk_ids,
                source_files=source_files,
                image_path=selected_image.image_path if use_image else None,
                image_caption=image_caption if use_image else None,
                review_terms=spec.get("review_terms"),
                review_refs_override=review_refs_override,
            )
        )

    if questions:
        return questions[:question_count]

    other_kps = [item for item in all_kps if item.kp_id != kp.kp_id]
    name_distractors = choose_distractors(kp.name, [item.name for item in all_kps], kp.kp_id + "-name")
    scenario_distractors = [scenario_for_kp(item) for item in other_kps]
    scenario_distractors = choose_distractors(scenario_for_kp(kp), dedupe_texts(scenario_distractors + [scenario_for_kp(kp)]), kp.kp_id + "-scenario")
    statement_distractors = [safe_snippet(item.description, 140) for item in other_kps]
    statement_distractors = choose_distractors(safe_snippet(kp.description, 140), dedupe_texts(statement_distractors + [safe_snippet(kp.description, 140)]), kp.kp_id + "-statement")
    note_distractors = [question_study_note(item) for item in other_kps]
    note_distractors = choose_distractors(question_study_note(kp), dedupe_texts(note_distractors + [question_study_note(kp)]), kp.kp_id + "-note")
    usecase_distractors = [f"{item.name} is most useful when {scenario_for_kp(item)[0].lower() + scenario_for_kp(item)[1:]}" for item in other_kps]
    correct_usecase = f"{kp.name} is most useful when {scenario_for_kp(kp)[0].lower() + scenario_for_kp(kp)[1:]}"
    usecase_distractors = choose_distractors(correct_usecase, dedupe_texts(usecase_distractors + [correct_usecase]), kp.kp_id + "-usecase")

    fallback_specs = [
        {
            "prompt": "Which knowledge point best matches the following study note?\n" + question_study_note(kp),
            "correct": kp.name,
            "distractors": name_distractors,
            "explanation": f"The study note describes {kp.name}: {clean_display_text(kp.description)}",
        },
        {
            "prompt": f"Which situation below is the best fit for {kp.name}?",
            "correct": scenario_for_kp(kp),
            "distractors": scenario_distractors,
            "explanation": f"This scenario directly reflects the role of {kp.name} in the lecture set.",
        },
        {
            "prompt": f"Which statement best summarizes {kp.name}?",
            "correct": safe_snippet(kp.description, 140),
            "distractors": statement_distractors,
            "explanation": f"The correct statement is the knowledge-point summary for {kp.name}.",
        },
        {
            "prompt": f"Which study note most likely belongs under {kp.name}?",
            "correct": question_study_note(kp),
            "distractors": note_distractors,
            "explanation": f"The correct study note combines the summary and cue words for {kp.name}.",
        },
        {
            "prompt": f"Which use case is the best fit for {kp.name}?",
            "correct": correct_usecase,
            "distractors": usecase_distractors,
            "explanation": f"The correct use case matches the lecture role of {kp.name}.",
        },
    ]
    for index, spec in enumerate(fallback_specs, start=1):
        questions.append(
            build_mcq_question(
                question_id=f"{kp.kp_id}-q{index}",
                kp=kp,
                prompt=str(spec["prompt"]),
                correct_text=str(spec["correct"]),
                distractors=[str(item) for item in spec["distractors"]],
                explanation=str(spec["explanation"]),
                review_chunks=review_chunks,
                source_chunk_ids=source_chunk_ids,
                source_files=source_files,
            )
        )
    return questions[:question_count]


def parse_json_response(raw_text: str) -> Any:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def resolve_llm_client() -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    if OpenAI is None:
        return None, None, "openai package is unavailable"

    api_key = os.getenv("COURSE_LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("KIMI_API_KEY")
    base_url = os.getenv("COURSE_LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("KIMI_BASE_URL")
    model = os.getenv("COURSE_LLM_MODEL") or os.getenv("DEEPSEEK_CHAT_MODEL") or os.getenv("KIMI_MODEL") or "deepseek-chat"

    if not api_key:
        return None, None, "No API key found in COURSE_LLM_API_KEY / DEEPSEEK_API_KEY / KIMI_API_KEY"

    try:
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        return client, model, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def llm_questions_for_kp(
    client: Any,
    model: str,
    kp: KnowledgePoint,
    support_chunks: Sequence[ChunkRecord],
    question_count: int,
) -> List[QuestionRecord]:
    support = "\n\n".join(
        f"[{chunk.relative_path} | chunk {chunk.chunk_index}] {safe_snippet(chunk.text, 360)}"
        for chunk in support_chunks[:4]
    )
    prompt = textwrap.dedent(
        f"""
        You are generating practice questions for a computer vision course.
        Use only the provided evidence.

        Knowledge point: {kp.name}
        Description: {kp.description}
        Keywords: {", ".join(kp.keywords)}

        Evidence:
        {support}

        Return strict JSON as a list with {question_count} items.
        Each item must contain:
        - question_type
        - question
        - answer
        - explanation
        - options (exactly 4 labeled options in the form "A. ...", "B. ...", "C. ...", "D. ...")
        - correct_option (one of A, B, C, D)

        Every question must be a multiple-choice question. Do not return short-answer or subjective questions.
        Prefer concept checks, method comparison, reasoning about algorithm steps, and small calculation-style prompts when the evidence supports them.
        Avoid questions that merely ask which keyword, which topic name, or which lecture label matches a sentence.
        Avoid trivia about file names, weeks, or slide numbers.
        """
    ).strip()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Generate grounded questions only. Output strict JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    parsed = parse_json_response(response.choices[0].message.content or "[]")
    results: List[QuestionRecord] = []
    source_files = sorted({chunk.relative_path for chunk in support_chunks})
    source_chunk_ids = [chunk.chunk_id for chunk in support_chunks]
    for index, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            continue
        results.append(
            QuestionRecord(
                question_id=f"{kp.kp_id}-llm-{index}",
                kp_id=kp.kp_id,
                kp_name=kp.name,
                question_type="multiple_choice",
                question=str(item.get("question", "")).strip(),
                answer=str(item.get("answer", "")).strip(),
                explanation=str(item.get("explanation", "")).strip(),
                options=[str(option) for option in item.get("options", []) if str(option).strip()],
                correct_option=(str(item["correct_option"]).strip() if item.get("correct_option") is not None else None),
                source_chunk_ids=source_chunk_ids,
                source_files=source_files,
                review_refs=build_review_refs(
                    str(item.get("question", "")).strip(),
                    str(item.get("answer", "")).strip(),
                    str(item.get("explanation", "")).strip(),
                    support_chunks,
                ),
            )
        )
    if not results:
        raise ValueError("LLM did not return valid question objects")
    return results[:question_count]


def generate_questions(
    knowledge_points: Sequence[KnowledgePoint],
    chunk_records: Sequence[ChunkRecord],
    slide_images: Sequence[SlideImageRecord],
    use_llm: bool,
    question_count: int,
) -> Tuple[List[QuestionRecord], Dict[str, Any]]:
    question_count = max(MIN_QUESTION_COUNT, question_count)
    chunk_map = {chunk.chunk_id: chunk for chunk in chunk_records}
    slide_images_by_doc_id: Dict[str, List[SlideImageRecord]] = defaultdict(list)
    slide_images_by_source: Dict[Tuple[str, int], SlideImageRecord] = {}
    for image in slide_images:
        slide_images_by_doc_id[image.doc_id].append(image)
        source_key = (normalized_relative_path(image.relative_path), int(image.slide_index))
        current = slide_images_by_source.get(source_key)
        if current is None or int(image.size_bytes) > int(current.size_bytes):
            slide_images_by_source[source_key] = image
    questions: List[QuestionRecord] = []
    llm_meta: Dict[str, Any] = {"enabled": False}

    client = None
    model = None
    if use_llm:
        client, model, error = resolve_llm_client()
        llm_meta = {
            "enabled": client is not None,
            "model": model,
            "error": error,
        }

    for kp in knowledge_points:
        if kp.name in QUIZ_EXCLUDED_KP_NAMES:
            continue
        support_chunks = [chunk_map[chunk_id] for chunk_id in kp.support_chunk_ids if chunk_id in chunk_map][:8]
        if not support_chunks:
            continue
        review_chunks = [chunk for chunk in chunk_records if chunk.relative_path in set(kp.source_files)] or list(support_chunks)
        fallback_generated = fallback_questions_for_kp(
            kp,
            support_chunks,
            review_chunks,
            chunk_records,
            knowledge_points,
            slide_images_by_doc_id,
            slide_images_by_source,
            question_count,
        )
        if QUESTION_BLUEPRINTS.get(kp.name):
            questions.extend(fallback_generated[:question_count])
            continue
        image_questions = [item for item in fallback_generated if item.image_path]
        llm_support_chunks = support_chunks[:4]
        if client is not None and model:
            last_exc: Optional[Exception] = None
            for attempt in range(3):
                try:
                    generated = llm_questions_for_kp(client, model, kp, llm_support_chunks, question_count)
                    if len(generated) >= question_count and all(item.question_type == "multiple_choice" for item in generated):
                        if image_questions:
                            merged = [image_questions[0]] + generated
                            questions.extend(merged[:question_count])
                        else:
                            questions.extend(generated[:question_count])
                        last_exc = None
                        break
                    raise ValueError("LLM output did not satisfy MCQ-only requirements")
                except Exception as exc:
                    last_exc = exc
                    if attempt < 2:
                        time.sleep(2 * (attempt + 1))
            if last_exc is None:
                continue
            llm_meta.setdefault("per_kp_errors", {})[kp.kp_id] = f"{type(last_exc).__name__}: {last_exc}"
        questions.extend(fallback_generated)

    return questions, llm_meta


def dense_index_exists(index_dir: Path) -> bool:
    return (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()


def dense_search_chunks(
    question: str,
    artifact_dir: Path,
    embedding_model: str,
    top_k: int,
) -> List[Tuple[float, Dict[str, Any]]]:
    if not dense_stack_available():
        return []
    index_dir = artifact_dir / "chunk_index"
    if not dense_index_exists(index_dir):
        return []
    try:
        embeddings = get_embeddings(embedding_model)
        vector_store = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        hits = vector_store.similarity_search_with_score(question, k=top_k)
        return [
            (
                float(score),
                {
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "relative_path": doc.metadata.get("relative_path"),
                    "week": doc.metadata.get("week"),
                    "title": doc.metadata.get("title"),
                    "source_type": doc.metadata.get("source_type"),
                    "unit_type": doc.metadata.get("unit_type"),
                    "unit_index": doc.metadata.get("unit_index"),
                    "text": doc.page_content,
                },
            )
            for doc, score in hits
        ]
    except Exception:
        return []


def build_pipeline(
    course_root: Path,
    artifact_dir: Path,
    no_llm: bool,
    lexical_only: bool,
    embedding_model: str,
    question_count: int,
    max_week: Optional[int] = None,
) -> Dict[str, Any]:
    question_count = max(MIN_QUESTION_COUNT, question_count)
    analyze_result = analyze_only(course_root, artifact_dir, max_week=max_week)
    raw_documents = [RawDocument(**row) for row in read_jsonl(artifact_dir / "raw_documents.jsonl")]
    chunk_records = build_chunk_records(raw_documents)
    knowledge_points = build_knowledge_points(chunk_records)
    chunk_map = {chunk.chunk_id: chunk for chunk in chunk_records}
    support_doc_ids = {
        chunk_map[chunk_id].doc_id
        for kp in knowledge_points
        for chunk_id in kp.support_chunk_ids
        if chunk_id in chunk_map
    }
    image_docs = [doc for doc in raw_documents if doc.doc_id in support_doc_ids]
    slide_images, slide_image_issues = extract_slide_images_for_docs(course_root, artifact_dir, image_docs)
    questions, llm_meta = generate_questions(
        knowledge_points=knowledge_points,
        chunk_records=chunk_records,
        slide_images=slide_images,
        use_llm=not no_llm,
        question_count=question_count,
    )

    write_jsonl(artifact_dir / "chunks.jsonl", (asdict(record) for record in chunk_records))
    write_json(artifact_dir / "knowledge_points.json", [asdict(kp) for kp in knowledge_points])
    write_json(artifact_dir / "questions.json", [asdict(question) for question in questions])
    write_json(artifact_dir / "slide_images.json", [asdict(image) for image in slide_images])

    chunk_index_meta = {
        "status": "skipped",
        "reason": "lexical-only build requested",
        "index_dir": str(artifact_dir / "chunk_index"),
        "count": len(chunk_records),
        "model_name": embedding_model,
    }
    kp_index_meta = {
        "status": "skipped",
        "reason": "lexical-only build requested",
        "index_dir": str(artifact_dir / "kp_index"),
        "count": len(knowledge_points),
        "model_name": embedding_model,
    }
    if not lexical_only:
        chunk_index_meta = build_chunk_dense_index(chunk_records, artifact_dir / "chunk_index", embedding_model)
        kp_index_meta = build_kp_dense_index(knowledge_points, artifact_dir / "kp_index", embedding_model)

    build_meta = {
        "course_root": str(course_root),
        "artifact_dir": str(artifact_dir),
        "max_week": max_week,
        "embedding_model": embedding_model,
        "llm_requested": not no_llm,
        "llm": llm_meta,
        "chunk_index": chunk_index_meta,
        "kp_index": kp_index_meta,
        "inventory": analyze_result["inventory"],
        "slide_image_count": len(slide_images),
        "slide_image_issues": slide_image_issues,
        "chunk_count": len(chunk_records),
        "knowledge_point_count": len(knowledge_points),
        "question_count": len(questions),
    }
    write_json(artifact_dir / "build_meta.json", build_meta)
    return build_meta


def format_ask_response(
    question: str,
    kp_hits: Sequence[Tuple[float, KnowledgePoint]],
    chunk_hits: Sequence[Tuple[float, ChunkRecord]],
) -> str:
    lines = [f"Question: {question}", "", "Matched knowledge points:"]
    if kp_hits:
        for score, kp in kp_hits:
            lines.append(f"- {kp.name} (score={score:.3f})")
            lines.append(f"  Description: {kp.description}")
    else:
        lines.append("- No knowledge point match found.")
    lines.extend(["", "Supporting chunks:"])
    if chunk_hits:
        for score, chunk in chunk_hits:
            lines.append(f"- [{chunk.relative_path} | chunk {chunk.chunk_index} | score={score:.3f}] {safe_snippet(chunk.text, 220)}")
    else:
        lines.append("- No chunk match found.")
    return "\n".join(lines)


def answer_question(
    artifact_dir: Path,
    question: str,
    top_k: int,
    lexical_only: bool,
    no_llm: bool,
    embedding_model: str,
) -> str:
    chunk_rows = read_jsonl(artifact_dir / "chunks.jsonl")
    kp_rows = read_json(artifact_dir / "knowledge_points.json")
    chunk_records = [ChunkRecord(**row) for row in chunk_rows]
    knowledge_points = [KnowledgePoint(**row) for row in kp_rows]

    kp_hits = lexical_rank(
        question,
        knowledge_points,
        text_getter=lambda kp: "\n".join([kp.name, kp.description, " ".join(kp.keywords)]),
        top_k=min(3, top_k),
    )

    if lexical_only:
        chunk_hits = lexical_rank(
            question,
            chunk_records,
            text_getter=lambda chunk: "\n".join([chunk.title, chunk.relative_path, chunk.text]),
            top_k=top_k,
        )
    else:
        dense_hits = dense_search_chunks(question, artifact_dir, embedding_model, top_k)
        if dense_hits:
            chunk_map = {chunk.chunk_id: chunk for chunk in chunk_records}
            chunk_hits = [
                (-score, chunk_map[item["chunk_id"]])
                for score, item in dense_hits
                if item["chunk_id"] in chunk_map
            ]
        else:
            chunk_hits = lexical_rank(
                question,
                chunk_records,
                text_getter=lambda chunk: "\n".join([chunk.title, chunk.relative_path, chunk.text]),
                top_k=top_k,
            )

    if not no_llm:
        client, model, _ = resolve_llm_client()
        if client is not None and model:
            evidence = "\n\n".join(
                f"[{chunk.relative_path} | chunk {chunk.chunk_index}] {safe_snippet(chunk.text, 320)}"
                for _, chunk in chunk_hits[: min(top_k, 5)]
            )
            prompt = textwrap.dedent(
                f"""
                Answer the question using only the provided course evidence.
                If the evidence is insufficient, say so explicitly.

                Question:
                {question}

                Matched knowledge points:
                {", ".join(kp.name for _, kp in kp_hits) or "None"}

                Evidence:
                {evidence or "None"}
                """
            ).strip()
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a grounded teaching assistant for a computer vision course."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=700,
                )
                answer = (response.choices[0].message.content or "").strip()
                if answer:
                    return answer
            except Exception:
                pass

    return format_ask_response(question, kp_hits, chunk_hits)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Course 4186 materials and build a standalone RAG pipeline.")
    parser.add_argument("--course-root", default=str(DEFAULT_COURSE_ROOT), help="Root folder of the course materials.")
    parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACT_DIR), help="Output folder for generated artifacts.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL, help="Embedding model for optional dense retrieval.")
    parser.add_argument("--max-week", type=int, default=None, help="Only include lecture and tutorial materials up to this week number.")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("analyze", help="Extract materials and generate inventory files.")

    build_parser_obj = subparsers.add_parser("build", help="Build chunks, knowledge points, and questions.")
    build_parser_obj.add_argument("--no-llm", action="store_true", help="Disable LLM-based question generation.")
    build_parser_obj.add_argument("--lexical-only", action="store_true", help="Skip dense FAISS indexes and use lexical retrieval only.")
    build_parser_obj.add_argument("--question-count", type=int, default=MIN_QUESTION_COUNT, help="Number of questions to generate per knowledge point.")

    ask_parser_obj = subparsers.add_parser("ask", help="Query the built artifacts.")
    ask_parser_obj.add_argument("--question", required=True, help="Question to ask against the built course knowledge base.")
    ask_parser_obj.add_argument("--top-k", type=int, default=5, help="Number of supporting chunks to retrieve.")
    ask_parser_obj.add_argument("--lexical-only", action="store_true", help="Use lexical retrieval only.")
    ask_parser_obj.add_argument("--no-llm", action="store_true", help="Disable LLM-based answer synthesis.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    course_root = Path(args.course_root)
    artifact_dir = Path(args.artifacts)

    if args.command == "analyze":
        result = analyze_only(course_root, artifact_dir, max_week=args.max_week)
        emit_output(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "build":
        build_meta = build_pipeline(
            course_root=course_root,
            artifact_dir=artifact_dir,
            no_llm=args.no_llm,
            lexical_only=args.lexical_only,
            embedding_model=args.embedding_model,
            question_count=max(MIN_QUESTION_COUNT, args.question_count),
            max_week=args.max_week,
        )
        emit_output(json.dumps(build_meta, indent=2, ensure_ascii=False))
        return

    if args.command == "ask":
        answer = answer_question(
            artifact_dir=artifact_dir,
            question=args.question,
            top_k=max(1, args.top_k),
            lexical_only=args.lexical_only,
            no_llm=args.no_llm,
            embedding_model=args.embedding_model,
        )
        emit_output(answer)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
