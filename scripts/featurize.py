# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Tuple

import polars as pl
import torch
from dvc import api as dvc_api
from torch import Tensor
from torch.utils.data import DataLoader

from product_recommendation.models.recommendation_dataset import RecommendationDataset


def featurize(
    categories_path: Path,
    item_properties_path: Path,
    sessions_path: Path,
    output_dir_path: Path,
) -> None:
    """Train models."""
    output_dir_path.parent.mkdir(parents=True, exist_ok=True)

    ldf_sessions = pl.scan_csv(sessions_path)
    ldf_item_properties = pl.scan_parquet(item_properties_path)
    ldf_categories = pl.scan_parquet(categories_path)

    ldf_item_properties_with_categories = (
        ldf_item_properties
            .join(
                ldf_categories,
                left_on="category_id",
                right_on="category_id",
                how="left",
            )
            .sort("timestamp")
    )

    (
        ldf_sessions
            .join_asof(
                ldf_item_properties_with_categories,
                on="timestamp",
                by="item_id",
                strategy="backward"
            )
            .collect()
            .write_parquet(output_dir_path / "enriched_sessions.parquet")
    )



if __name__ == "__main__":
    params = dvc_api.params_show(stages="featurize").get("featurize")
    if params is None:
        raise RuntimeError("Missing DVC's params.yaml?")

    featurize(
        Path("categories_path"),
        Path("item_properties_path"),
        Path("sessions_path"),
        Path("output_dir_path"),
    )
