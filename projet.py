import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data.csv')

temp = x.sub(x.mean())
x_scaled = temp.div(x.std())

pca = PCA(n_components=3)
pca.fit(x_scaled)

pca_res = pca.fit_transform(x_scaled)


//4
biplot(score=pca_res[:,0:2],
       coeff=np.transpose(pca.components_[0:2, :]),
       cat=y,
       cmap="viridis",
       coeff_labels=list(X.columns))
plt.show()