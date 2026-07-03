# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import polars as pl
from dvc import api as dvc_api

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
from scripts.utils import update_metadata  # noqa: E402


def prepare_categories(category_tree_path: Path, output_dir_path: Path) -> None:
    """Prepare categories from category tree."""
    print(f"Creating path: {output_dir_path.resolve()}")
    output_dir_path.mkdir(parents=True, exist_ok=True)

    ldf_categories = pl.scan_csv(category_tree_path)

    def build_ancestors_path(category_id: pl.Int32) -> list[int]:
        ancestors = []

        current_category = category_id
        current_parent = (
            ldf_categories.filter(pl.col("categoryid").eq(current_category))
            .select("parentid")
            .collect()
        )
        while not current_parent.is_empty():
            parent = current_parent.item()
            if parent is None:
                break

            ancestors.append(parent)
            current_category = parent
            current_parent = (
                ldf_categories.filter(pl.col("categoryid") == current_category)
                .select("parentid")
                .collect()
            )

        return ancestors + [category_id]

    categories_metadata = ldf_categories.select(
        min_category_id=pl.col("categoryid").min(),
        max_category_id=pl.col("categoryid").max(),
        num_categories=pl.col("categoryid").n_unique(),
    ).collect()
    metadata = {
        "num_categories": categories_metadata["num_categories"].item(),
        "min_category_id": categories_metadata["min_category_id"].item(),
        "max_category_id": categories_metadata["max_category_id"].item(),
    }
    update_metadata(
        existing_metadata_path=output_dir_path / "categories_metadata.json",
        new_metadata=metadata,
    )

    (
        ldf_categories.with_columns(
            path=pl.col("categoryid").map_elements(
                build_ancestors_path, return_dtype=pl.List(pl.Int32)
            )
        )
        .rename({"categoryid": "category_id"})
        .select(["category_id", "path"])
        .collect()
        .write_parquet(output_dir_path / "categories.parquet")
    )

    # df_category_tree = (
    #     ldf_categories.with_columns(
    #         ancestor_path=pl.col("categoryid").map_elements(
    #             build_ancestors_path, return_dtype=pl.List(pl.Int32)
    #         )
    #     )
    #     .select(
    #         category_id=pl.col("categoryid"),
    #         ancestor_path=pl.col("ancestor_path"),
    #         ancestor_path_reversed=pl.col("ancestor_path").list.reverse(),
    #     )
    #     .collect()
    # )

    # max_depth = df_category_tree.select(
    #     pl.col("ancestor_path_reversed").list.len().max().alias("max_depth")
    # ).item()

    # (
    #     df_category_tree.with_columns(
    #         pl.col("ancestor_path_reversed")
    #         .list.concat(pl.lit([0] * max_depth))
    #         .list.slice(0, max_depth)
    #         .alias("padded_path")
    #     )
    #     .with_columns(
    #         [
    #             pl.col("padded_path").list.get(i).alias(f"parent_id_{i + 1}")
    #             for i in range(max_depth)
    #         ]
    #     )
    #     .drop(["padded_path", "ancestor_path", "ancestor_path_reversed"])
    #     .write_parquet(output_dir_path / "categories.parquet")
    # )


if __name__ == "__main__":
    params = dvc_api.params_show(stages="prepare_categories").get("prepare_categories")
    if params is None:
        raise RuntimeError("Missing DVC's params.yaml?")

    prepare_categories(
        Path(params["category_tree_path"]),
        Path(params["output_dir_path"]),
    )
