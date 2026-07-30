from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    classification_report,
    normalized_mutual_info_score,
    silhouette_score,
    roc_auc_score,
    confusion_matrix,
)

# ==================================================================
# 1. Load Saved Embeddings
# ==================================================================
embeddings_path = Path.cwd() / "bert_embeddings.npz"
if not embeddings_path.exists():
    raise FileNotFoundError(
        f"Could not find '{embeddings_path}'. Run extract_embeddings.py first!"
    )

print(f"Loading embeddings from {embeddings_path}...")
data = np.load(embeddings_path)
X_bert = data["embeddings"]
labels = data["labels"]
n_samples = len(labels)

print(f"Loaded embeddings shape: {X_bert.shape}")
print(f"Class distribution: {np.bincount(labels)}")

# ==================================================================
# 2. PCA Dimensionality Reduction (for visualization only)
# ==================================================================
print("\nApplying PCA (2 Components)...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_bert)

var_explained = pca.explained_variance_ratio_
print(
    f"PCA Explained Variance Ratio: PC1 = {var_explained[0]:.2%}, "
    f"PC2 = {var_explained[1]:.2%} (Total: {sum(var_explained):.2%})"
)

# ==================================================================
# 3. K-Means Clustering (Unsupervised Baseline)
# ==================================================================
print("\nRunning K-Means Clustering (k=2)...")
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_bert)

# ==================================================================
# 4. Unsupervised Performance Metrics
# ==================================================================
print("\n" + "=" * 50)
print("          UNSUPERVISED CLUSTERING METRICS         ")
print("=" * 50)

# Robust label alignment (works for any binary encoding)
cluster_0_majority = np.bincount(labels[cluster_labels == 0]).argmax()
cluster_1_majority = np.bincount(labels[cluster_labels == 1]).argmax()

mapped_cluster_labels = np.empty_like(cluster_labels)
mapped_cluster_labels[cluster_labels == 0] = cluster_0_majority
mapped_cluster_labels[cluster_labels == 1] = cluster_1_majority

print("\n--- Classification Metrics (Mapped Clusters) ---")
print(
    classification_report(
        labels, mapped_cluster_labels, target_names=["Legitimate", "Spam"]
    )
)

print("--- Clustering Quality Metrics ---")
print(f"Silhouette Score:          {silhouette_score(X_bert, cluster_labels):.4f}")
print(f"Adjusted Rand Index (ARI): {adjusted_rand_score(labels, cluster_labels):.4f}")
print(
    f"Normalized Mutual Info:    {normalized_mutual_info_score(labels, cluster_labels):.4f}"
)

# ==================================================================
# 5. Supervised Cross-Validation Evaluation
# ==================================================================
print("\n" + "=" * 50)
print("      SUPERVISED CROSS-VALIDATION (Logistic)      ")
print("=" * 50)

# Scale embeddings for linear classifier convergence
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_bert)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")

# Out-of-fold probability predictions (unbiased estimates)
y_proba = cross_val_predict(lr, X_scaled, labels, cv=cv, method="predict_proba")[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

print("\n--- Cross-Validation Classification Report ---")
print(classification_report(labels, y_pred, target_names=["Legitimate", "Spam"]))
print(f"ROC-AUC Score: {roc_auc_score(labels, y_proba):.4f}")

# Confusion Matrix
cm = confusion_matrix(labels, y_pred)
print(f"\n--- Confusion Matrix ---")
print(f"                 Predicted")
print(f"                 Legit  Spam")
print(f"Actual Legit    {cm[0,0]:5d}  {cm[0,1]:5d}")
print(f"       Spam     {cm[1,0]:5d}  {cm[1,1]:5d}")

# ==================================================================
# 6. Decision Boundary Triage
# ==================================================================
print("\n" + "=" * 50)
print("          DECISION BOUNDARY TRIAGE                ")
print("=" * 50)

inbox = np.sum(y_proba < 0.30)
quarantine = np.sum((y_proba >= 0.30) & (y_proba <= 0.75))
spam = np.sum(y_proba > 0.75)

print("\n--- Decision Boundary Triage Distribution ---")
print(f"Inbox------(P < 0.30):-------{inbox:6d} ({inbox / n_samples:.2%})")
print(f"Quarantine-(0.30 <= P <= 0.75): {quarantine:6d} ({quarantine / n_samples:.2%})")
print(f"Spam-------(P > 0.75):-------{spam:6d} ({spam / n_samples:.2%})")

# Extra insight: triage breakdown by ground truth
triage_df = pd.DataFrame(
    {
        "true_label": np.where(labels == 0, "Legitimate", "Spam"),
        "triage": pd.cut(
            y_proba,
            bins=[-0.01, 0.30, 0.75, 1.01],
            labels=["Inbox", "Quarantine", "Spam"],
        ),
    }
)
print("\n--- Triage Breakdown by True Label ---")
print(
    triage_df.groupby(["triage", "true_label"], observed=False)
    .size()
    .unstack(fill_value=0)
)
print("=" * 50)

# ==================================================================
# 7. Visualization
# ==================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Ground Truth Labels in PCA Space
scatter1 = axes[0].scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels,
    cmap="coolwarm",
    alpha=0.5,
    s=15,
    edgecolors="none",
)
axes[0].set_title("BERT Embeddings (PCA 2D) - True Labels", fontsize=14)
axes[0].set_xlabel(f"PC1 ({var_explained[0]:.1%} Variance)")
axes[0].set_ylabel(f"PC2 ({var_explained[1]:.1%} Variance)")
cbar1 = fig.colorbar(scatter1, ax=axes[0], ticks=[0, 1])
cbar1.ax.set_yticklabels(["Legitimate (0)", "Spam (1)"])

# Plot 2: K-Means Clustering in PCA Space
scatter2 = axes[1].scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
    cmap="viridis",
    alpha=0.5,
    s=15,
    edgecolors="none",
)

# Transform K-Means centroids into PCA space for plotting
centroids_pca = pca.transform(kmeans.cluster_centers_)
axes[1].scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    c="red",
    marker="X",
    s=200,
    linewidths=2,
    edgecolors="black",
    label="Cluster Centroids",
)

axes[1].set_title("BERT Embeddings (PCA 2D) - K-Means Clusters", fontsize=14)
axes[1].set_xlabel(f"PC1 ({var_explained[0]:.1%} Variance)")
axes[1].set_ylabel(f"PC2 ({var_explained[1]:.1%} Variance)")
axes[1].legend()
cbar2 = fig.colorbar(scatter2, ax=axes[1], ticks=[0, 1])
cbar2.ax.set_yticklabels(["Cluster 0", "Cluster 1"])

plt.tight_layout()
output_plot = "bert_pca_clustering.png"
plt.savefig(output_plot, dpi=300)
print(f"\nVisualization saved successfully as '{output_plot}'.")
plt.show()
