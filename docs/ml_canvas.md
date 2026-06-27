# ML Canvas - Recommendation System

**Objective**

- Provide personalized product recommendations based on user browsing behavior (views, add-to-carts, transactions) to increase engagement and conversion.
- Task: Recommendation (Ranking/Top-K).

**Success metric**

- **ML Metrics:**
  - Primary: Hit Rate @ 10 (HR@10) >= 0.15.
  - Secondary: Normalized Discounted Cumulative Gain (NDCG@10) >= 0.10.
  - Baseline: MRR (Mean Reciprocal Rank).
- **Business KPI:**
  - Increase click-through rate (CTR) on recommended items by 10%.
  - Increase conversion rate from "view" to "add-to-cart" for recommended items.

**Data**

- [RetailRocket eCommerce Dataset (Kaggle)](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).
  - `events.csv`: Behavioral data (view, addtocart, transaction).
  - `category_tree.csv`: Taxonomy of product categories.
  - `item_properties.csv`: Attributes of products (brand, availability, etc.).
- Interactions: ~2.7M events, ~1.4M users, ~235K items.

**Features**

- User latent embeddings (learned from interaction history).
- Item latent embeddings (learned from interaction history).
- Interaction type weight (transaction > addtocart > view).
- Temporal features: Recency of interactions, time of day.
- Item metadata: Category ID, property values.

**Model**

- **Baseline:** Popularity-based or simple Collaborative Filtering (Matrix Factorization via Sklearn/Surprise).
- **Primary:** PyTorch Embedding-based Model (Neural Collaborative Filtering or Two-Tower architecture).

**Constraints**

- Must follow Clean Code standards (SOLID, type hints).
- Containerized deployment (Docker).
- Scalable serving via Kubernetes (FastAPI).
- Inference latency < 150ms for Top-10 recommendations.

**Risks**

- **Cold Start:** New users or items with no history.
- **Data Sparsity:** Most user-item pairs have no interaction.
- **Popularity Bias:** Model might over-recommend "trending" items, ignoring the long tail.
- **Concept Drift:** Seasonal trends (e.g., Black Friday) changing user behavior.
- **Performance Leakage:** Using future events to predict past interactions during training.
