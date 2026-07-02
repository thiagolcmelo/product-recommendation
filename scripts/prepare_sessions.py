# -*- coding: utf-8 -*-
from json import dump
from pathlib import Path

import polars as pl
from dvc import api as dvc_api


def prepare_sessions(events_path: Path, output_dir_path: Path) -> None:
    """Prepare sessions from events."""
    print(f"Creating path: {output_dir_path.resolve()}")
    output_dir_path.mkdir(parents=True, exist_ok=True)

    events = {
        "view": 1,
        "addtocart": 2,
        "transaction": 3,
    }

    with open(output_dir_path / "event_ids.json", "w") as f:
        dump(events, f, indent=4)

    ldf_events = pl.scan_csv(events_path)

    (
        ldf_events.sort(["visitorid", "timestamp"])
        .with_columns(
            time_diff=pl.col("timestamp").diff().over("visitorid"),
            event_id=pl.col("event").replace_strict(events),
        )
        .with_columns(
            is_new_session=(
                (pl.col("time_diff").is_null()) | (pl.col("time_diff") > 1_800_000)
            ).cast(pl.Int32)
        )
        .with_columns(session_id=pl.col("is_new_session").cum_sum().over("visitorid"))
        .with_columns(
            session_event_id=pl.col("timestamp")
            .rank()
            .over(["visitorid", "session_id"])
        )
        .with_columns(
            # history=pl.concat_list("timestamp", "itemid", "event_id")
            # .implode()
            # .over(["visitorid", "session_id"])
            # .list.head(pl.col("session_event_id") - 1)
            history=pl.col("itemid")
            .implode()
            .over(["visitorid", "session_id"])
            .list.head(pl.col("session_event_id") - 1)
        )
        .filter(pl.col("history").list.len() > 0)
        .select(
            timestamp=pl.col("timestamp"),
            visitor_id=pl.col("visitorid"),
            event_id=pl.col("event_id"),
            item_id=pl.col("itemid"),
            transaction_id=pl.col("transactionid"),
            history=pl.col("history"),
        )
        .collect()
        .write_parquet(output_dir_path / "sessions.parquet")
    )


if __name__ == "__main__":
    params = dvc_api.params_show(stages="prepare_sessions").get("prepare_sessions")
    if params is None:
        raise RuntimeError("Missing DVC's params.yaml?")

    prepare_sessions(
        Path(params["events_path"]),
        Path(params["output_dir_path"]),
    )
