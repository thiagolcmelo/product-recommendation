# -*- coding: utf-8 -*-
import sys
from enum import Enum
from pathlib import Path

import mlflow
import polars as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from product_recommendation.models.item_tower import ItemTower  # type: ignore[import-untyped]
from product_recommendation.models.recommendation_dataset import RecommendationDataset  # type: ignore[import-untyped]
from product_recommendation.models.session_tower import SessionTower  # type: ignore[import-untyped]
from product_recommendation.models.two_tower_recommender import TwoTowerRecommender  # type: ignore[import-untyped]

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
from scripts.utils import load_metadata  # noqa: E402


class ModelType(str, Enum):
    BASELINE = "baseline"
    MAIN = "main"


def get_model_type_mode(model_type: ModelType) -> str:
    if model_type == ModelType.BASELINE:
        return "pooling"
    elif model_type == ModelType.MAIN:
        return "gru"
    else:
        raise ValueError(
            f"Invalid model_type '{model_type}'. Must be one of {[e.value for e in ModelType]}"
        )


def recommendation_collate_fn(
    batch: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    batch: A list of tuples coming from your Dataset's __getitem__
    Each tuple contains: (history, target_item, category_path, property_ids, availability)
    """
    # 1. Unzip the batch into separate lists
    histories, target_items, category_paths, property_ids_list, availabilities = zip(
        *batch
    )

    # 2. Stack the features that are already a fixed size
    # shape: [batch_size, max_seq_len]
    hist_tensor = torch.stack(histories)
    # shape: [batch_size]
    target_tensor = torch.stack(target_items)
    # shape: [batch_size]
    avail_tensor = torch.stack(availabilities)

    # 3. Dynamically pad the variable-length lists for this batch
    # batch_first=True makes the shape: [batch_size, max_elements_in_this_batch]
    cat_tensor = torch.nn.utils.rnn.pad_sequence(
        category_paths,  # type: ignore[arg-type]
        batch_first=True,
        padding_value=0,
    )
    prop_tensor = torch.nn.utils.rnn.pad_sequence(
        property_ids_list,  # type: ignore[arg-type]
        batch_first=True,
        padding_value=0,
    )

    return hist_tensor, target_tensor, cat_tensor, prop_tensor, avail_tensor


def train_models(
    enriched_sessions_path: Path,
    metadata_path: Path,
    output_dir_path: Path,
    model_type: ModelType,
    model_version: str,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    output_dim: int,
) -> None:
    """Train models."""
    print(f"Creating path: {output_dir_path.resolve()}")
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Lazy loading path: {enriched_sessions_path.resolve()}")
    df_enriched_sessions = pl.read_parquet(enriched_sessions_path)

    # Instantiate your dataset
    dataset = RecommendationDataset(df_enriched_sessions)

    # Create the data loader loader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=recommendation_collate_fn,  # Fixes variable-length properties in batches
    )

    # Define Catalog Sizes (Vocab sizes) for Embedding layers
    # These numbers should match the maximum ID + 1 from your preprocessing mapping
    metadata = load_metadata(metadata_path)
    num_items = (
        metadata.get("item_properties_metadata", {}).get("max_item_id", 466866) + 1
    )
    num_properties = (
        metadata.get("item_properties_metadata", {}).get("max_property_id", 1104) + 1
    )
    num_categories = (
        metadata.get("categories_metadata", {}).get("max_category_id", 466866) + 1
    )

    # Initialize Towers and Master Model
    session_tower = SessionTower(
        vocab_size=num_items,
        embedding_dim=output_dim,
        hidden_dim=64,
        mode=get_model_type_mode(model_type),
    )
    item_tower = ItemTower(
        num_items=num_items,
        num_categories=num_categories,
        num_properties=num_properties,
        output_dim=output_dim,
    )

    # Enable system metrics logging
    mlflow.enable_system_metrics_logging()

    # Training with MLflow logging
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(
            {
                "model_type": model_type.value,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "output_dim": output_dim,
                "num_items": num_items,
                "num_properties": num_properties,
                "num_categories": num_categories,
            }
        )

        model = TwoTowerRecommender(session_tower, item_tower)

        # Force execution to stay on your CPU
        device = torch.device("cpu")
        model.to(device)

        # Define Optimization Tools
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # BCEWithLogitsLoss expects positive similarities to trend toward 1, negative to 0
        loss_fn = nn.BCEWithLogitsLoss()

        # Execution Loop
        for epoch in range(num_epochs):
            model.train()  # Sets network layers to training mode
            total_loss = 0.0

            for batch in train_loader:
                # Move raw data tensors over to the CPU
                hist, pos_item, cat, prop, avail = [
                    tensor.to(device) for tensor in batch
                ]
                # --- Positive Pass ---
                # Generate similarity scores for the actual event item
                pos_scores = model(hist, pos_item, cat, prop, avail)

                # --- Negative Sampling Pass ---
                # Randomly generate dummy target item IDs to simulate unclicked items
                neg_item = torch.randint(
                    1, num_items, size=pos_item.shape, device=device
                )
                # Note: In a production run, look up properties for these random items too!
                neg_scores = model(hist, neg_item, cat, prop, avail)

                # Assemble Targets: 1 for authentic interactions, 0 for random samples
                scores = torch.cat([pos_scores, neg_scores])
                targets = torch.cat(
                    [torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]
                )

                # --- Backpropagation Magic ---
                loss = loss_fn(scores, targets)

                optimizer.zero_grad()  # Erase stale memory from the previous loop iteration
                loss.backward()  # Calculate gradients across both towers
                optimizer.step()  # Tweak embedding parameters and GRU weights

                total_loss += loss.item()

            print(
                f"Epoch {epoch + 1}/{num_epochs} complete. Average Loss: {total_loss / len(train_loader):.4f}"
            )
            mlflow.log_metrics(
                {"train_loss": total_loss / len(train_loader)}, step=epoch
            )

        model_name = (
            f"two_tower_recommender_{model_type.value.lower()}_v{model_version}"
        )

        mlflow.pytorch.log_model(model, name=model_name)

        torch.save(
            model.state_dict(),
            output_dir_path / f"{model_name}.pkl",
        )
