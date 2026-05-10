import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("gearbox_k12.csv")

X      = df[["vib_hs", "vib_ls", "speed"]].values
labels = df["cluster"].values

# ── PCA 2-D projection ────────────────────────────────────────────────────────
pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis",
                      alpha=0.6, s=5)
plt.colorbar(scatter, label="Cluster ID")
plt.title("Gearbox K-Means (k=12) – PCA 2-D projection")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.tight_layout()
plt.savefig("gearbox_clusters_pca.png", dpi=120)
plt.show()
print("Saved → gearbox_clusters_pca.png")

# ── Direct 3-D scatter ────────────────────────────────────────────────────────
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax  = fig.add_subplot(111, projection="3d")
sc  = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels,
                 cmap="tab10", s=2, alpha=0.5)
plt.colorbar(sc, ax=ax, label="Cluster ID")
ax.set_xlabel("vib_hs (norm)")
ax.set_ylabel("vib_ls (norm)")
ax.set_zlabel("speed (norm)")
ax.set_title("Gearbox K-Means (k=12) – 3-D scatter")
plt.tight_layout()
plt.savefig("gearbox_clusters_3d.png", dpi=120)
plt.show()
print("Saved → gearbox_clusters_3d.png")
