# Copyright (c) mrmilbe

from pathlib import Path

HEADER = "# Copyright (c) mrmilbe"
EXCLUDED_DIR = Path("WBoxVenv")

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    files = (root / ".." / "..")
    files.resolve()
    for path in root.rglob("*.py"):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if EXCLUDED_DIR in rel.parents:
            continue
        text = path.read_text(encoding="utf-8")
        if text.lstrip().startswith(HEADER):
            continue
        path.write_text(f"{HEADER}\n\n{text}", encoding="utf-8")
