import pandas as pd  
import numpy  as np
import matplotlib.pyplot as plt 
import conjuntosEVT
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.preprocessing import StandardScaler

dataframe = pd.read_csv('Data/pooled_outputs.csv')

dataframe_ = dataframe.drop(columns=['class'])


#Estandarización
"""
scaler = StandardScaler()

X_train = scaler.fit_transform(conjuntosEVT.train_f)
X_test = scaler.transform(conjuntosEVT.test_f)

#LDA

lda = LDA()

LDA_train = lda.fit_transform(X_train, conjuntosEVT.train_c)

LDA_test = lda.transform(X_test)

print(np.array(LDA_train).shape)
print(np.array(LDA_test).shape)



print(lda.explained_variance_ratio_)


plt.figure(figsize=(10, 6))


class_0 = LDA_train[conjuntosEVT.train_c == 0]
class_1 = LDA_train[conjuntosEVT.train_c == 1]
plt.scatter(class_0, np.zeros_like(class_0), color='blue', label='Clase 0')
plt.scatter(class_1, np.zeros_like(class_1), color='red', label='Clase 1')

plt.show()
"""
#--------------------------------
scaler2 = StandardScaler()

X_all = scaler2.fit_transform(dataframe_)

lda2 = LDA()

LDA_all = lda2.fit_transform(X_all, dataframe['class'].values)

print(np.array(LDA_all).shape)

print(lda2.explained_variance_ratio_)