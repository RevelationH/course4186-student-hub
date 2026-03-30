from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pythoncom
    import win32com.client
except Exception as exc:  # pragma: no cover
    pythoncom = None
    win32com = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


PDF_FORMAT = 32
POWERPOINT_SUFFIXES = {".ppt", ".pptx"}


def iter_missing_powerpoint_files(root: Path) -> list[Path]:
    rows: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in POWERPOINT_SUFFIXES:
            continue
        if path.with_suffix(".pdf").exists():
            continue
        rows.append(path)
    return rows


def convert_one(presentation_app, source_path: Path) -> tuple[bool, str]:
    target_path = source_path.with_suffix(".pdf")
    if target_path.exists():
        return True, "already_exists"

    presentation = None
    try:
        presentation = presentation_app.Presentations.Open(
            str(source_path),
            WithWindow=False,
            ReadOnly=True,
        )
        presentation.SaveAs(str(target_path), PDF_FORMAT)
        return (target_path.exists(), "converted" if target_path.exists() else "missing_output")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        if presentation is not None:
            try:
                presentation.Close()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert missing Course 4186 PPT/PPTX materials to PDF.")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[2] / "4186" / "4186"),
        help="Root folder containing the course materials.",
    )
    args = parser.parse_args()

    if IMPORT_ERROR is not None or pythoncom is None or win32com is None:
        print(f"ERROR: pywin32 is unavailable: {IMPORT_ERROR}", file=sys.stderr)
        return 1

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"ERROR: material root does not exist: {root}", file=sys.stderr)
        return 1

    missing = iter_missing_powerpoint_files(root)
    print(f"Material root: {root}")
    print(f"Missing PDFs: {len(missing)}")
    if not missing:
        print("Nothing to convert.")
        return 0

    pythoncom.CoInitialize()
    app = None
    converted = 0
    failed: list[tuple[Path, str]] = []
    try:
        app = win32com.client.DispatchEx("PowerPoint.Application")
        app.Visible = 1
        for index, path in enumerate(missing, start=1):
            ok, status = convert_one(app, path)
            relative = path.relative_to(root)
            if ok:
                converted += 1
                print(f"[{index}/{len(missing)}] OK    {relative} -> {path.with_suffix('.pdf').name} ({status})")
            else:
                failed.append((path, status))
                print(f"[{index}/{len(missing)}] FAIL  {relative} ({status})")
    finally:
        if app is not None:
            try:
                app.Quit()
            except Exception:
                pass
        pythoncom.CoUninitialize()

    print(f"Converted: {converted}")
    print(f"Failed: {len(failed)}")
    if failed:
        print("Failed files:")
        for path, status in failed:
            try:
                relative = path.relative_to(root)
            except Exception:
                relative = path
            print(f"- {relative}: {status}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
