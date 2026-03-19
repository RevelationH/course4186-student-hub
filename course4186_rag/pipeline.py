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

from pypdf import PdfReader
from question_blueprints import QUESTION_BLUEPRINTS

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

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


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COURSE_ROOT = Path(os.getenv("COURSE4186_COURSE_ROOT", r"D:\digital_human\4186\4186"))
DEFAULT_ARTIFACT_DIR = SCRIPT_DIR / "artifacts_week1_week6"
DEFAULT_MAX_WEEK = int(os.getenv("COURSE4186_MAX_WEEK", "6"))
DEFAULT_EMBED_MODEL = os.getenv("COURSE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".ppt"}
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


def normalize_text(text: str) -> str:
    lines: List[str] = []
    for raw_line in text.replace("\x00", " ").splitlines():
        line = WS_RE.sub(" ", raw_line).strip()
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
        if path.is_file() and path.name != ".DS_Store"
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
    "\u0ddc": "",
    "\u0ddd": "",
    "\U0001d465": "x",
}


def clean_display_text(text: str) -> str:
    cleaned = str(text or "")
    for source, target in DISPLAY_REPLACEMENTS.items():
        cleaned = cleaned.replace(source, target)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


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

    records: List[SlideImageRecord] = []
    issues: List[Dict[str, str]] = []
    for relative_path_str, slides in sorted(docs_by_file.items()):
        file_path = course_root / Path(relative_path_str)
        if not file_path.exists():
            issues.append({"relative_path": relative_path_str, "issue": "PPTX file not found during image extraction"})
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
                    deck_slug = slugify(str(Path(relative_path_str).with_suffix("")))
                    extension = Path(image_name).suffix.lower()
                    output_name = f"slide-{slide_index:03d}-{short_hash(f'{relative_path_str}|{slide_index}|{image_name}|{size_bytes}')}{extension}"
                    relative_image_path = Path("images") / deck_slug / output_name
                    absolute_image_path = artifact_dir / relative_image_path
                    ensure_dir(absolute_image_path.parent)
                    absolute_image_path.write_bytes(blob)

                    records.append(
                        SlideImageRecord(
                            doc_id=doc.doc_id,
                            relative_path=relative_path_str,
                            slide_index=slide_index,
                            image_path=relative_image_path.as_posix(),
                            image_caption=f"Lecture figure from {Path(relative_path_str).as_posix()} slide {slide_index}",
                            image_name=image_name,
                            size_bytes=size_bytes,
                        )
                    )
        except Exception as exc:
            issues.append({"relative_path": relative_path_str, "issue": f"{type(exc).__name__}: {exc}"})

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
            "image understanding",
            "image representation",
            "recognition",
            "scene understanding",
            "vision tasks",
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
        "keywords": ["image resampling", "resampling", "sampling", "resolution"],
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
        "keywords": ["stereo vision", "depth map", "stereo reconstruction", "correspondence"],
    },
    {
        "name": "Epipolar geometry",
        "start_week": 10,
        "description": "Constraining stereo correspondences with epipolar lines and geometric relations.",
        "keywords": ["epipolar", "epipolar constraint", "cross product", "essential matrix", "fundamental matrix"],
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
        "keywords": ["motion", "optical flow", "moving world"],
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
        scored: List[Tuple[int, ChunkRecord]] = []
        for chunk in chunk_records:
            blob = "\n".join([chunk.title, chunk.relative_path, chunk.text]).lower()
            score = 0
            for keyword in entry["keywords"]:
                term = keyword.lower()
                if term in blob:
                    score += 2 if term in chunk.title.lower() or term in chunk.relative_path.lower() else 1
            if score > 0:
                scored.append((score, chunk))
        if not scored:
            continue

        scored.sort(key=lambda pair: (-pair[0], pair[1].relative_path, pair[1].chunk_index))
        selected = [chunk for _, chunk in scored[:8]]
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


def build_mcq_question(
    question_id: str,
    kp: KnowledgePoint,
    prompt: str,
    correct_text: str,
    distractors: Sequence[str],
    explanation: str,
    source_chunk_ids: Sequence[str],
    source_files: Sequence[str],
    image_path: Optional[str] = None,
    image_caption: Optional[str] = None,
) -> QuestionRecord:
    options, correct_option = build_option_lines(correct_text, distractors, question_id)
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
        image_path=image_path,
        image_caption=clean_display_text(image_caption) if image_caption else None,
    )


def select_question_image(
    support_chunks: Sequence[ChunkRecord],
    slide_images_by_doc_id: Dict[str, List[SlideImageRecord]],
) -> Optional[SlideImageRecord]:
    candidates: List[SlideImageRecord] = []
    seen_paths: set[str] = set()
    for chunk in support_chunks:
        for image in slide_images_by_doc_id.get(chunk.doc_id, []):
            if image.image_path in seen_paths:
                continue
            seen_paths.add(image.image_path)
            candidates.append(image)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item.size_bytes, item.relative_path, item.slide_index))
    return candidates[0]


def fallback_questions_for_kp(
    kp: KnowledgePoint,
    support_chunks: Sequence[ChunkRecord],
    all_kps: Sequence[KnowledgePoint],
    slide_images_by_doc_id: Dict[str, List[SlideImageRecord]],
    question_count: int,
) -> List[QuestionRecord]:
    question_count = max(MIN_QUESTION_COUNT, question_count)
    source_files = sorted({chunk.relative_path for chunk in support_chunks})
    source_chunk_ids = [chunk.chunk_id for chunk in support_chunks]
    selected_image = select_question_image(support_chunks, slide_images_by_doc_id)
    specs = QUESTION_BLUEPRINTS.get(kp.name, [])
    questions: List[QuestionRecord] = []
    for index, spec in enumerate(specs[:question_count], start=1):
        use_image = bool(spec.get("use_image")) and selected_image is not None
        questions.append(
            build_mcq_question(
                question_id=f"{kp.kp_id}-q{index}",
                kp=kp,
                prompt=str(spec.get("prompt", "")).strip(),
                correct_text=str(spec.get("correct", "")).strip(),
                distractors=[str(item).strip() for item in spec.get("distractors", [])],
                explanation=str(spec.get("explanation", "")).strip(),
                source_chunk_ids=source_chunk_ids,
                source_files=source_files,
                image_path=selected_image.image_path if use_image else None,
                image_caption=selected_image.image_caption if use_image else None,
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
            "prompt": f"Which revision note most likely belongs under {kp.name}?",
            "correct": question_study_note(kp),
            "distractors": note_distractors,
            "explanation": f"The correct revision note combines the summary and cue words for {kp.name}.",
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
    for image in slide_images:
        slide_images_by_doc_id[image.doc_id].append(image)
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
        support_chunks = [chunk_map[chunk_id] for chunk_id in kp.support_chunk_ids if chunk_id in chunk_map][:4]
        if not support_chunks:
            continue
        fallback_generated = fallback_questions_for_kp(kp, support_chunks, knowledge_points, slide_images_by_doc_id, question_count)
        image_questions = [item for item in fallback_generated if item.image_path]
        if client is not None and model:
            last_exc: Optional[Exception] = None
            for attempt in range(3):
                try:
                    generated = llm_questions_for_kp(client, model, kp, support_chunks, question_count)
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
    parser.add_argument("--max-week", type=int, default=DEFAULT_MAX_WEEK, help="Only include lecture and tutorial materials up to this week number.")

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
