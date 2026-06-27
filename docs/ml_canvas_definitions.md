# ML Canvas Definitions & Core Concepts

This document provides a detailed breakdown of the terminology, metrics, and methodology expectations outlined in the Recommendation System's ML Canvas.

---

## 1. Core Concepts: What does "Latent" mean?

In machine learning, **latent** means **hidden** or **not directly observable**. 

*   **Explicit Features:** These are features we can see directly in our tables. For example: `category_id = 12`, `price = 25.00`, or `interaction_type = "addtocart"`.
*   **Latent Features (Embeddings):** These are abstract concepts that the neural network learns on its own by analyzing patterns of behavior.

### The Analogy
Imagine you are recommending movies: 
*   **Explicit features** are genres: "Action", "Comedy", "Sci-Fi".
*   **Latent features** are hidden dimensions of taste that don't have simple names. For example, the model might find a hidden dimension that captures *"nostalgic 80s synth-wave aesthetic"* or *"slow-burn psychological tension with a twist ending"*. 

Instead of manual cataloging, the PyTorch model converts every User and Item into a vector of numbers (e.g., 64 floating-point numbers). This vector is the **latent embedding**. If User A's embedding vector is mathematically close to Item B's embedding vector in this 64-dimensional space, it means they share similar hidden taste patterns, and we should recommend that item.

---

## 2. Recommendation Metrics Demystified

When evaluating a recommendation system offline, we use ranking-based metrics. Here is how they operate:

### Hit Rate @ 10 (HR@10)
*   **Concept:** Did the user's actual favorite/interacted item make it into our top 10 recommended list?
*   **How it works:** For a test user, we know they bought Item X. We ask the model to recommend 10 items.
    *   If Item X is in the top 10 $\rightarrow$ **Hit (1)**
    *   If Item X is NOT in the top 10 $\rightarrow$ **Miss (0)**
*   **HR@10** is the percentage of hits across all test users. A target of `0.15` means that for 15% of our users, we successfully predicted their next interaction within our top 10 recommendations.

### NDCG @ 10 (Normalized Discounted Cumulative Gain)
*   **Concept:** It is not just about *if* the correct item is in the top 10; it is about *where* it is. An item recommended at #1 is much better than an item recommended at #10.
*   **How it works:**
    *   **Gain:** The relevance of the item (e.g., 1 if clicked, 0 if not).
    *   **Discount:** We divide the gain by a logarithmic penalty based on its position (rank). Rank 1 has almost no discount; rank 10 has a heavy discount.
    *   **Normalized:** We scale the score between `0` and `1` so we can average it across different users.
*   An NDCG of `0.10` indicates that we are successfully ranking highly relevant items near the top of our lists.

### MRR (Mean Reciprocal Rank)
*   **Concept:** How far down the list do users have to look to find the *first* relevant recommendation?
*   **How it works:** If the first correct recommendation is at rank $r$, the score is $1/r$.
    *   If it's at rank 1, score = $1/1 = 1.0$
    *   If it's at rank 2, score = $1/2 = 0.5$
    *   If it's at rank 10, score = $1/10 = 0.1$
*   We take the average of these reciprocal ranks across all users.

---

## 3. Establishing Expectation Thresholds

In a production-ready ML project, target values (like a 10% CTR increase or HR@10 >= 0.15) are established through a rigorous baseline and business alignment process:

1.  **Establishing the Performance Floor (Baselines):**
    Before building a complex PyTorch neural network, we run simple baseline models on our dataset. For example, a **Popularity Baseline** (always recommend the top 10 most viewed items) or a **Simple Matrix Factorization**.
    *   If the Popularity Baseline gets a `Hit Rate @ 10` of `0.07`, we know our minimum baseline is 7%.
    *   We then set our neural network target reasonably higher (e.g., `0.15` or 15%), representing a significant, double-the-baseline improvement.

2.  **Defining Business KPIs (e.g., +10% CTR):**
    These metrics are determined by Product Managers and Finance teams using:
    *   **Financial Viability:** *"We calculated that the engineering cost of building and maintaining this GPU/K8s infrastructure requires a minimum 5% lift in total conversion rate to break even."*
    *   **Industry Benchmarks:** E-commerce companies typically see a 5% to 15% increase in Click-Through Rates (CTR) when transitioning from static or popularity-based widgets to personalized deep-learning-based recommendations.

3.  **A/B Testing and Shadow Deployments:**
    We don't truly *know* if we reached the +10% business KPI during training.
    *   During development, we focus on **offline metrics** (HR@10, NDCG).
    *   Once deployed, we run an **A/B test** where Group A sees no recommendations (or popular recommendations) and Group B sees our PyTorch recommendations. Only then do we measure the actual online business metric (e.g., CTR and conversion lift).
