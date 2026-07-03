# -*- coding: utf-8 -*-
import sys
from pathlib import Path

from dvc import api as dvc_api

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
from scripts.train_models import ModelType, train_models  # noqa: E402


if __name__ == "__main__":
    params = dvc_api.params_show(stages="train_baseline_model").get(
        "train_baseline_model"
    )
    if params is None:
        raise RuntimeError("Missing DVC's params.yaml?")

    train_models(
        Path(params["enriched_sessions_path"]),
        Path(params["metadata_path"]),
        Path(params["output_dir_path"]),
        model_type=ModelType.BASELINE,
        model_version=params["model_version"],
        batch_size=int(params["batch_size"]),
        learning_rate=float(params["learning_rate"]),
        num_epochs=int(params["num_epochs"]),
        output_dim=int(params["output_dim"]),
    )
