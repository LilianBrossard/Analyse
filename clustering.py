import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


# ==========================
# Chargement des données
# ==========================
df = pd.read_csv("output.csv")

num_cols = [
    "core_shots", "core_goals", "core_saves", "core_assists", "core_score",
    "core_shooting_percentage", "positioning_avg_distance_to_ball",
    "demo_inflicted", "demo_taken"
]
X = df[num_cols].fillna(0)

X_scaled = StandardScaler().fit_transform(X)

# ==========================
# Méthode du coude (inertie)
# ==========================
K = range(2, 10)
inertias = []
for k in K:
    model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(6, 5))
plt.plot(K, inertias, marker='o')
plt.title("Rocket League - Méthode du coude")
plt.xlabel("Nombre de clusters")
plt.ylabel("Inertie")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("rl_inertie.png")
plt.close()

# ==========================
# Clustering avec K choisi
# ==========================
k = 4
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centers_pca = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(8, 6))
for i in range(k):
    idx = labels == i
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Cluster {i+1}", alpha=0.5)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
            c="red", marker="X", s=200, label="Centres")
plt.title(f"Rocket League - Clustering K-Means (k={k})")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.savefig("rl_clustering_pca.png")
plt.close()

print(" Graphiques générés : rl_inertie.png, rl_clustering_pca.png")

# ==========================
# CAH avec linkage ward
# ==========================
Z = linkage(X_scaled, method="ward")

# Dendrogramme
plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode="level", p=4, leaf_rotation=90, leaf_font_size=10)
plt.title("Rocket League - Dendrogramme (CAH)")
plt.xlabel("Joueurs")
plt.ylabel("Distance")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("rl_dendrogramme.png")
plt.close()

# ==========================
# Méthode du plus grand saut
# ==========================
distances = Z[:, 2]  
diffs = np.diff(distances)  
max_gap_idx = np.argmax(diffs)  
seuil = (distances[max_gap_idx] + distances[max_gap_idx+1]) / 2

# Attribution des classes
labels = fcluster(Z, t=seuil, criterion="distance")
nb_classes = len(np.unique(labels))

print(f" Nombre de classes trouvées avec la méthode du 'plus grand saut' : {nb_classes}")
print(" Graphique généré : rl_dendrogramme.png")

# ==========================
# Visualisation PCA des clusters CAH
# ==========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for i in range(1, nb_classes + 1):
    idx = labels == i
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Cluster {i}", alpha=0.6)
plt.title(f"Rocket League - Clustering CAH (Ward) ({nb_classes} classes)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.savefig("rl_cah_clusters.png")
plt.close()

print(" Graphique généré : rl_cah_clusters.png")

