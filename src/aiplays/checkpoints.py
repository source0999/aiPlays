from __future__ import annotations

import os
from pathlib import Path


def newest_checkpoint(directory: Path) -> Path | None:
    candidates = [
        path for path in directory.glob("*.zip") if path.is_file() and path.stat().st_size > 0
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def update_latest(source: Path, models_dir: Path) -> Path:
    latest = models_dir / "latest.zip"
    temp = latest.with_suffix(".tmp")
    temp.write_bytes(source.read_bytes())
    os.replace(temp, latest)
    return latest
