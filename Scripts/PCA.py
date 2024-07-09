from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataframe = pd.read_csv('Data/standar_datapack.csv')
clases = dataframe['class']
dataframe_ = dataframe.drop(columns=['class'])

pca = PCA(n_components=0.95)
pca.fit(dataframe_)

principalComponents  = pca.transform(dataframe_)

print(len(principalComponents[0]))


principalDf = pd.DataFrame(data = principalComponents)

finalDf = pd.concat([principalDf, clases], axis = 1)

print(finalDf.describe())


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['class'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 0]
               , finalDf.loc[indicesToKeep, 32]
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

plt.show()