import pandas as pd  
import numpy  as np
import matplotlib.pyplot as plt 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


from sklearn.preprocessing import StandardScaler

dataframe = pd.read_csv('Data/pooled_outputs.csv')

dataframe_ = dataframe.drop(columns=['class'])


#Estandarización
#--------------------------------
scaler = StandardScaler()

X_all = scaler.fit_transform(dataframe_)

lda2 = LDA()

LDA_all = lda2.fit_transform(X_all, dataframe['class'].values)

print(np.array(LDA_all).shape)

print(lda2.explained_variance_ratio_)

plt.figure(figsize=(12,8))

class_0 = LDA_all[dataframe['class'].values == 0]
class_1 = LDA_all[dataframe['class'].values == 1]
plt.scatter(class_0, np.zeros_like(class_0), color='blue', label='Clase 0')
plt.scatter(class_1, np.zeros_like(class_1), color='red', label='Clase 1')

plt.show()