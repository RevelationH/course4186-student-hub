from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path


def normalized_key(stem: str) -> str:
    text = re.sub(r"\(\d+\)", " ", str(stem or "").lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def canonical_rank(path: Path) -> tuple[int, int, str]:
    stem = path.stem
    paren_count = len(re.findall(r"\(\d+\)", stem))
    return (paren_count, len(stem), stem.lower())


def canonical_pdf_map(material_root: Path) -> dict[Path, Path]:
    chosen: dict[Path, Path] = {}
    for folder in [p for p in material_root.rglob("*") if p.is_dir()] + [material_root]:
        pdfs = sorted(folder.glob("*.pdf"))
        buckets: dict[str, list[Path]] = defaultdict(list)
        for pdf in pdfs:
            buckets[normalized_key(pdf.stem)].append(pdf)
        for items in buckets.values():
            if not items:
                continue
            canonical = min(items, key=canonical_rank)
            relative = canonical.relative_to(material_root)
            chosen[relative] = canonical
    return chosen


def sync_materials(material_root: Path, bundle_root: Path) -> tuple[int, int]:
    destination_root = bundle_root / "course4186_materials"
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    mapping = canonical_pdf_map(material_root)
    copied = 0
    for relative, source in sorted(mapping.items()):
        target = destination_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    return copied, len(mapping)


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy canonical Course 4186 lecture PDFs into the GitHub upload bundle.")
    parser.add_argument(
        "--materials-root",
        default=str(Path(__file__).resolve().parents[2] / "4186" / "4186"),
        help="Root folder containing the original lecture materials.",
    )
    parser.add_argument(
        "--bundle-root",
        default=str(Path(__file__).resolve().parents[2] / "course4186_github_upload_20260319"),
        help="Root folder of the GitHub upload bundle.",
    )
    args = parser.parse_args()

    material_root = Path(args.materials_root).resolve()
    bundle_root = Path(args.bundle_root).resolve()
    copied, canonical_count = sync_materials(material_root, bundle_root)
    print(f"Material root: {material_root}")
    print(f"Bundle root: {bundle_root}")
    print(f"Canonical PDFs copied: {copied}")
    print(f"Canonical PDF entries: {canonical_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
