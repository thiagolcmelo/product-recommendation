# -*- coding: utf-8 -*-

import json
from pathlib import Path


def update_metadata(existing_metadata_path: Path, new_metadata: dict) -> None:
    """Update existing metadata JSON file with new metadata."""
    if existing_metadata_path.exists():
        with open(existing_metadata_path, "r") as f:
            existing_metadata = json.load(f)
        existing_metadata.update(new_metadata)
        new_metadata = existing_metadata

    with open(existing_metadata_path, "w") as f:
        json.dump(new_metadata, f, indent=4)


def load_metadata(metadata_path: Path) -> dict:
    """Load metadata from JSON file."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata
