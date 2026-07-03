# -*- coding: utf-8 -*-
import json
import sys
from pathlib import Path

import polars as pl
from dvc import api as dvc_api

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
from scripts.utils import update_metadata  # noqa: E402


def merge_sessions_with_item_properties(
    categories_path: Path,
    item_properties_path: Path,
    sessions_path: Path,
    output_dir_path: Path,
) -> None:
    """Merge sessions with item properties."""
    print(f"Creating path: {output_dir_path.resolve()}")
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Lazy loading path: {sessions_path.resolve()}")
    ldf_sessions = pl.scan_parquet(sessions_path)
    print(f"Lazy loading path: {item_properties_path.resolve()}")
    ldf_item_properties = pl.scan_parquet(item_properties_path)
    print(f"Lazy loading path: {categories_path.resolve()}")
    ldf_categories = pl.scan_parquet(categories_path)

    ldf_item_properties_with_categories = ldf_item_properties.select(
        pl.all().sort_by("timestamp").over("item_id")
    ).join(
        ldf_categories,
        left_on="category_id",
        right_on="category_id",
        how="left",
    )

    (
        ldf_sessions.select(pl.all().sort_by("timestamp").over("visitor_id"))
        .join_asof(
            ldf_item_properties_with_categories,
            on="timestamp",
            by="item_id",
            strategy="backward",
        )
        .fill_null(0)
        .rename({"item_id": "target"})
        .collect()
        .write_parquet(output_dir_path / "enriched_sessions.parquet")
    )


def merge_metadata(
    categories_metadata_path: Path,
    item_properties_metadata_path: Path,
    output_metadata_path: Path,
) -> None:
    """Merge metadata from categories and item properties."""

    with open(categories_metadata_path, "r") as f:
        categories_metadata = json.load(f)

    with open(item_properties_metadata_path, "r") as f:
        item_properties_metadata = json.load(f)

    merged_metadata = {
        "categories_metadata": {**categories_metadata},
        "item_properties_metadata": {**item_properties_metadata},
    }

    update_metadata(
        existing_metadata_path=output_metadata_path,
        new_metadata=merged_metadata,
    )


def featurize(
    categories_path: Path,
    item_properties_path: Path,
    sessions_path: Path,
    output_dir_path: Path,
    categories_metadata_path: Path,
    item_properties_metadata_path: Path,
) -> None:
    """Featurize sessions with item properties and categories."""
    merge_sessions_with_item_properties(
        categories_path,
        item_properties_path,
        sessions_path,
        output_dir_path,
    )

    merge_metadata(
        categories_metadata_path,
        item_properties_metadata_path,
        output_dir_path / "metadata.json",
    )


if __name__ == "__main__":
    params = dvc_api.params_show(stages="featurize").get("featurize")
    if params is None:
        raise RuntimeError("Missing DVC's params.yaml?")

    featurize(
        Path(params["categories_path"]),
        Path(params["item_properties_path"]),
        Path(params["sessions_path"]),
        Path(params["output_dir_path"]),
        Path(params["categories_metadata_path"]),
        Path(params["item_properties_metadata_path"]),
    )
