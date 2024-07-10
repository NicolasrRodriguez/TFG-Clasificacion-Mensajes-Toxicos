from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import conjuntosEVT

from sklearn.preprocessing import StandardScaler

#Estandarización

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

class_0 = pC_train[conjuntosEVT.train_c == 0]
class_1 = pC_train[conjuntosEVT.train_c == 1]
plt.scatter(class_0, np.zeros_like(class_0), color='blue', label='Clase 0')
plt.scatter(class_1, np.zeros_like(class_1), color='red', label='Clase 1')

plt.show()