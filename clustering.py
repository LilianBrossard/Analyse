import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from itertools import cycle
import matplotlib.cm as cm

df = pd.read_csv("output.csv")
num_cols = [
    "core_shots", "core_goals", "core_saves", "core_assists", "core_score",
    "core_shooting_percentage", "positioning_avg_distance_to_ball",
    "demo_inflicted", "demo_taken"
]
X = df[num_cols].fillna(0)
X_scaled = StandardScaler().fit_transform(X)

def run_plot_kmeans(n_clusters, ax):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_scaled)
    y_kmeans = kmeans.predict(X_scaled)
    centers = kmeans.cluster_centers_

    labels_dict = {i: f"Cluster {i+1}" for i in range(n_clusters)}
    colors = cycle(cm.tab10.colors)

    for i in range(n_clusters):
        color = next(colors)
        idx = y_kmeans == i
        ax.scatter(X_scaled[idx, 0], X_scaled[idx, 1], color=color, s=50, label=labels_dict[i], alpha=0.25)
        ax.scatter(centers[i, 0], centers[i, 1], edgecolors="k", linewidth=2, color=color, s=200, alpha=1)
    ax.legend()

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

K_range = range(2, 10)
inertias = []
for k in K_range:
    model = KMeans(n_clusters=k).fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(6,5))
plt.plot(K_range, inertias, marker='o')
plt.title("Rocket League - Méthode du coude")
plt.xlabel("Nombre de clusters")
plt.ylabel("Inertie")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("rl_inertie.png")
plt.close()

fig, ax = plt.subplots(figsize=(8,6))
run_plot_kmeans(3, ax)
plt.title("Rocket League - K-Means Clustering")
plt.savefig("rl_kmeans_clusters.png")
plt.close()

model_cah = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(X_scaled)

plt.figure(figsize=(12,6))
plot_dendrogram(model_cah, truncate_mode="level", p=4)
plt.title("Rocket League - Dendrogramme (CAH)")
plt.xlabel("Nombre de points dans la classe")
plt.ylabel("Distance")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("rl_dendrogramme.png")
plt.close()

print("Graphiques générés : rl_inertie.png, rl_kmeans_clusters.png, rl_dendrogramme.png")
