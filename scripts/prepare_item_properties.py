# -*- coding: utf-8 -*-
from pathlib import Path

import polars as pl
from dvc import api as dvc_api


def prepare_item_properties(
    item_properties_part1_path: Path,
    item_properties_part2_path: Path,
    output_dir_path: Path,
) -> None:
    """Prepare item properties."""
    output_dir_path.parent.mkdir(parents=True, exist_ok=True)

    ldf_item_properties = pl.concat(
        [
            pl.scan_csv(item_properties_part1_path),
            pl.scan_csv(item_properties_part2_path),
        ]
    )

    df_encoded_values = pl.concat(
        [
            ldf_item_properties.filter(pl.col("property").is_in(["available"]))
            .with_columns(
                property=pl.col("property"),
                value_original=pl.col("value"),
                value_encoded=pl.col("value").cast(pl.UInt32, strict=False),
            )
            .select(["property", "value_encoded", "value_original"])
            .unique()
            .sort(["property", "value_encoded"]),
            ldf_item_properties.filter(pl.col("property").is_in(["categoryid"]))
            .with_columns(
                property=pl.col("property"),
                value_original=pl.col("value"),
                value_encoded=pl.col("value").cast(pl.UInt32, strict=False),
            )
            .select(["property", "value_encoded", "value_original"])
            .unique()
            .sort(["property", "value_encoded"]),
            ldf_item_properties.filter(
                ~pl.col("property").is_in(["categoryid", "available"])
            )
            .with_columns(
                property=pl.col("property"),
                value_original=pl.col("value"),
                value_encoded=pl.col("value")
                .rank("dense")
                .over("property")
                .cast(pl.UInt32),
            )
            .select(["property", "value_encoded", "value_original"])
            .unique()
            .sort(["property", "value_encoded"]),
        ]
    )
    df_encoded_values.collect().write_parquet(
        output_dir_path / "item_properties_values_encoded.parquet"
    )

    total_mingled_properties = (
        ldf_item_properties.filter(
            ~pl.col("property").is_in(["categoryid", "available"])
        )
        .select(pl.col("property").unique().cast(pl.Int32, strict=False).len())
        .collect()
        .item()
    )

    (
        ldf_item_properties.join(
            df_encoded_values,
            left_on=["property", "value"],
            right_on=["property", "value_original"],
            how="left",
        )
        .sort(["itemid", "timestamp"])
        .pivot(
            on="property",
            index=["itemid", "timestamp"],
            values="value_encoded",
            on_columns=["available", "categoryid"]
            + [str(i) for i in range(total_mingled_properties)],
        )
        .rename({"itemid": "item_id", "categoryid": "category_id"})
        .with_columns(
            available=pl.col("available").fill_null(2),
            category_id=pl.col("category_id").fill_null(-1),
        )
        .fill_null(0)
        .rename({str(i): f"property_{i}" for i in range(total_mingled_properties)})
        .collect()
        .write_parquet(output_dir_path / "item_properties.parquet")
    )


if __name__ == "__main__":
    params = dvc_api.params_show(stages="prepare_item_properties").get(
        "prepare_item_properties"
    )
    if params is None:
        raise RuntimeError("Missing DVC's params.yaml?")

    prepare_item_properties(
        Path(params["item_properties_part1_path"]),
        Path(params["item_properties_part2_path"]),
        Path(params["output_dir_path"]),
    )
