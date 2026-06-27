# Model Card Core Concepts & Definitions

This document details the critical concepts, terminology, and engineering methodologies mentioned in the Recommendation System's Model Card.

---

## 1. Implicit vs. Explicit Feedback

In recommendation systems, we must decide how we learn what a user likes. There are two primary categories of behavioral signals:

### Explicit Feedback
*   **What it is:** The user directly states their preference (e.g., rating a movie 5 stars, writing a review, or clicking "Thumbs Up/Down").
*   **The Problem:** It is incredibly rare. Less than 1% of users actually rate products explicitly. It is also highly biased (people tend to only rate things they absolutely love or absolutely hate, leaving a massive empty middle).

### Implicit Feedback (Our Choice for RetailRocket)
*   **What it is:** The user's natural, unprompted behavior (e.g., viewing an item, searching, adding a product to a cart, or finalizing a purchase).
*   **The Advantage:** It is abundant! Every single interaction is captured as data.
*   **The Challenge:** It is noisy. If a user views an item, do they actually like it? Or did they click it by accident? Or did they view it, hate it, and immediately leave?
*   **Our Solution:** We assign **weights (strengths)** to different actions to model preference intensity:
    $$\text{View (Weight: 1)} \rightarrow \text{Add-to-Cart (Weight: 5)} \rightarrow \text{Transaction (Weight: 10)}$$
    The PyTorch neural network learns to predict these weighted interactions, treating a transaction as a much stronger latent preference than a simple view.

---

## 2. Temporal Splits vs. Random Splits (Preventing Data Leakage)

When training standard classifiers, we do a random stratified split (e.g., 80% train, 20% test). **In recommendation systems, a random split is a disaster called "Data Leakage."**

### The Leakage Scenario
If we randomly split interactions, the training set might contain a user buying a product on Friday, and the test set might contain that same user viewing it on Thursday. The model will look "backwards" in time, easily predicting the Thursday view because it already knows the Friday purchase happened. Your offline accuracy will look amazing (e.g., 99%), but in production, the model will fail because it cannot see into the future.

### The Solution: Temporal Split
We choose a specific "cutoff time" (e.g., October 1st):
*   **Training Set:** All behavioral events *before* October 1st.
*   **Test Set:** All behavioral events *on or after* October 1st.
This perfectly mimics how the system operates in production: using past history to predict future actions.

---

## 3. Popularity Bias & Feedback Loops

This is the most common failure point in recommendation engineering.

### Popularity Bias
Standard machine learning algorithms want to minimize mathematical loss. Recommending the top 5 most popular items on the entire site (e.g., iPhones) is a highly safe bet. The model will quickly learn to over-index on popularity, recommendation lists will become identical for every user, and the "personalized" system is ruined.

### The Feedback Loop (Echo Chambers)
This is a self-reinforcing circle:
1.  The model recommends a popular item at position #1.
2.  Because it's at position #1, users click it more.
3.  The model gets retrained on new click data, sees that the item got tons of clicks, and concludes: *"I was right! This item is incredibly popular!"*
4.  It recommends it even more aggressively, burying niche or long-tail items forever.

### Mitigation
We track this in our Model Card and Monitoring Plan using a **Diversity Metric** (like the Gini Coefficient). In Phase 2, we can implement exploration strategies (like $\epsilon$-greedy exploration) or popularity-penalization factors to ensure the long tail gets recommended.

---

## 4. Ethical Considerations: Filter Bubbles & Privacy

### Filter Bubbles (Echo Chambers)
If a user clicks on three sci-fi books, and the model starts recommending *only* sci-fi books, the user is locked in a bubble. In e-commerce, this limits product discovery and reduces average order value over time.

### Data Privacy (Anonymization)
We train models on `user_id` and `item_id`. If we use raw database keys (e.g., `user_id = 148902` which links to "Thiago Melo"), an attacker who hacks the model weights or intercepts API calls could reconstruct Thiago's entire private browsing history.
*   **Best Practice:** We must use anonymized, hashed, or sequentially mapped IDs that exist only within the ML pipeline context, keeping real user identities completely separated.
