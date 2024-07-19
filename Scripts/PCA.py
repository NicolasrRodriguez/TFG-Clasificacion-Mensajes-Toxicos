from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import conjuntosEVT

from sklearn.preprocessing import StandardScaler


dataframe = pd.read_csv('Data/pooled_outputs.csv')

dataframe_ = dataframe.drop(columns=['class'])

#Estandarización
"""
scaler = StandardScaler()


X_train = scaler.fit_transform(conjuntosEVT.train_f)
X_test = scaler.transform(conjuntosEVT.test_f)

#Se establece el 95%
pca = PCA(n_components=0.95)

#Se emtrena con el conjunto de emtrenamiento

pC_train = pca.fit_transform(X_train)

pC_test = pca.transform(X_test)


print(np.array(pC_train).shape)
print(np.array(pC_test).shape)

"""
#------------------------todo-----------------

scaler2 = StandardScaler()

X_all = scaler2.fit_transform(dataframe_)


pca2 = PCA(n_components=0.95)

pC_all = pca2.fit_transform(X_all)

print(np.array(pC_all).shape)

