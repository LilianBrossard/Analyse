import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('output.csv')

#3 ACP
temp = x.sub(x.mean())
x_scaled = temp.div(x.std())

pca=PCA(n_components=3)
pca.fit(x_scaled)

pca_res = pca.fit_transform(x_scaled)

#3.1
eig = pd.DataFrame({
    "Dimension": [f"Dim{str(x+1)}" for x in range(6)],
    "Valeur propre": pca.explained_variance_,
    "% valeur propre":
    np.round(pca.explained_variance_ratio_*100),
    "% cum. valeur prop.":
    np.round(np.cumsum(pca.explained_variance_ratio_)*100)
})

y1 = list(pca.explained_variance_ratio_)
x1 = range(len(y1))
plt.bar(x1, y1)
plt.show()

# # Standardisation 
# # La standardisation (aussi appeléé normalisation) consiste à soustraire la moyenne et diviser par l'écart-type
# # La distribution des données est ainsi centrée réduite (moyenne = 0, écart-type = 1)

# X = df(df.columns[0:4])
# y = df["nom"]
# temp = X.sub(X.mean())
# X_scaled = temp.div(X.std())

# # Entrainement du modèle
# # Nous appelons X_scaled le jeu de valeur standardisées au quel nous appliquerons l'ACP. Dans l'exercice [...]
# # le nombre de composantes conservées est fixé à 3, vous changerez cette valeur en fonction de vos besoins.

# n_components = 3
# pca = PCA(n_components=n_components)
# pca.fit(X_scaled)

# # pca.fit permet d'obtenir la modélisation de l'ACP. Afin d'obtenir les résultats pour les individus, il [...]
# #fonctions fit() et transform() de l'ACP. Une autre possibilité est d'utiliser la fonction fit_transform() réalisé automatiquement la combinaison des deux.

# pca_res = pca.fit_transform(X_scaled)

# # Calcul des valeurs propres
# # Parmi l'ensemble des valeurs de la PCA vous retrouverez :
# # - Les valeurs propres des composantes : pca.singular_values_
# # - Les pourcentages de variance expliq

# # - ration de variance expliquée cumulée