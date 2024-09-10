from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


dataframe = pd.read_csv('Data/pooled_outputs.csv')

dataframe_ = dataframe.drop(columns=['class'])

#------------------------

scaler2 = StandardScaler()

X_all = scaler2.fit_transform(dataframe_)


pca2 = PCA(n_components=0.95)

pC_all = pca2.fit_transform(X_all)

print(np.array(pC_all).shape)

