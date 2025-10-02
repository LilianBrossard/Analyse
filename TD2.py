import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from mca import MCA


data = pd.read_csv('output.csv')
data = data[["color", "team_region", "player_tag", "winner", "car_name"]]

dc = pd.DataFrame(pd.get_dummies(data[["color", "team_region", "player_tag", "winner", "car_name"]]))
dc.head()

mcaFic = MCA(dc, benzecri=False)

plt.scatter(mcaFic.fs_c()[ :, 0], mcaFic.fs_c()[ :, 1])
for i, j, nom in zip(mcaFic.fs_c()[ :, 0], mcaFic.fs_c()[ :, 1], dc.columns):
    plt.text(i, j, nom)
plt.show()
