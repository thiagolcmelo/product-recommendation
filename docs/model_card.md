# Model Card — Recommendation Embedding Model

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | `rec_embedding_v1` |
| **Architecture** | Embedding-based (Two-Tower or NCF) with learned latent representations |
| **Framework** | PyTorch 2.x |
| **Training date** | TBD (Phase 2) |
| **Version** | 1.0.0 |
| **Contact** | Thiago Melo |

## Intended Use

- **Primary use**: Predict top-K product recommendations for e-commerce users based on implicit feedback history.
- **Target users**: E-commerce platform integration (API).
- **Out of scope**:
  - Content-based filtering solely on text descriptions (not the primary focus).
  - Real-time session-based recommendations without historical user ID (initial version focus is on identified users).

## Training Data

| Field | Value |
|-------|-------|
| **Dataset** | RetailRocket eCommerce Dataset |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) |
| **Size** | ~2.7M events |
| **Feedback Type** | Implicit (view=1, addtocart=2, transaction=3) |
| **Split** | Temporal split (Train: past, Test: most recent period) |

## Performance Metrics

| Metric | **PyTorch Embedding** | Matrix Factorization | Popularity Baseline |
|--------|----------------------:|---------------------:|--------------------:|
| Hit Rate @ 10 | TBD | TBD | TBD |
| NDCG @ 10 | TBD | TBD | TBD |
| MRR | TBD | TBD | TBD |

*(Metrics to be populated after model training phase)*

## Limitations

- **Sparsity**: High ratio of users/items to interactions may lead to poor embeddings for low-activity entities.
- **Static Item Properties**: Item properties change over time (e.g., availability); the model needs a mechanism to handle dynamic updates.
- **No Textual/Visual Context**: Initial version relies on IDs and categories, not image or description analysis.
- **Evaluation Loop**: Offline metrics (HR@10) may not perfectly correlate with online business KPIs (Conversion Rate).

## Biases

- **Popularity Bias**: Heavy-tail items may never be recommended.
- **Feedback Loop**: Users only interact with what they see; if the system only shows popular items, it reinforces their popularity.
- **Geographic/Cultural**: Behavioral patterns in the dataset may reflect specific market characteristics not applicable globally.

## Ethical Considerations

- **Filter Bubbles**: Over-personalization might limit user exposure to diverse products.
- **Privacy**: User history must be handled according to GDPR/LGPD; use of hashed IDs is mandatory.
- **Manipulation**: Avoid optimizing purely for "time-on-site" if it leads to addictive browsing patterns.

## Maintenance

| Field | Value |
|-------|-------|
| **Retraining cadence** | Weekly or Daily (depending on traffic volume) |
| **Drift monitoring** | PSI on user/item activity levels |
| **Owner** | Thiago Melo |
