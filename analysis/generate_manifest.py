from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "package_inventory.csv"
MANIFEST = ROOT / "MANIFEST.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def category(relative_path: str) -> str:
    if relative_path.startswith("source/"):
        return "source_document"
    if relative_path.startswith("data/tables/"):
        return "paper_table"
    if relative_path.startswith("data/behavior/"):
        return "behavior_data"
    if relative_path.startswith("data/materials/"):
        return "material_pair_data"
    if relative_path.startswith("configs/"):
        return "experiment_config"
    if relative_path.startswith("data/representation/"):
        return "representation_data"
    if relative_path.startswith("analysis/"):
        return "validation_script"
    if relative_path.startswith("checks/"):
        return "validation_output"
    if relative_path.startswith("outputs/"):
        return "workbook"
    if relative_path.lower().endswith(".md"):
        return "documentation"
    return "package_metadata"


def payload_files() -> list[Path]:
    excluded = {INVENTORY.resolve(), MANIFEST.resolve()}
    return sorted(
        (
            path
            for path in ROOT.rglob("*")
            if path.is_file() and path.resolve() not in excluded
        ),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def main() -> None:
    with INVENTORY.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "size_bytes", "sha256", "category"])
        for path in payload_files():
            relative_path = path.relative_to(ROOT).as_posix()
            writer.writerow(
                [relative_path, path.stat().st_size, sha256(path), category(relative_path)]
            )

    files_for_manifest = sorted(
        (path for path in ROOT.rglob("*") if path.is_file() and path.resolve() != MANIFEST.resolve()),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    with MANIFEST.open("w", encoding="utf-8", newline="\n") as handle:
        for path in files_for_manifest:
            handle.write(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n")

    print(
        {
            "status": "pass",
            "inventory_rows": len(payload_files()),
            "manifest_rows": len(files_for_manifest),
        }
    )


if __name__ == "__main__":
    main()
