from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly as ply

from sklearn.decomposition import PCA



dataframe = pd.read_csv('Data/pooled_outputs.csv')

df_without_last_column = dataframe.drop(columns=['class'])


clf = IsolationForest(contamination = 0.35 , random_state = 42) 


#anomalias en clase 0: mensajes negativos

embeddings_negativos = dataframe[dataframe['class'] == 0]

neg_embeddings_array = np.array(embeddings_negativos.values.tolist())

embeddings_negativos['is_anomaly'] = clf.fit_predict(neg_embeddings_array)

print(embeddings_negativos['is_anomaly'].any() == -1)

#anomalias en clase 1: mensajes positivos

embeddings_positivos = dataframe[dataframe['class'] == 1]

pos_embeddings_array = np.array(embeddings_positivos.values.tolist())

embeddings_positivos['is_anomaly'] = clf.fit_predict(pos_embeddings_array)

print(embeddings_positivos['is_anomaly'].any == -1)

#visualización
"""
dims = 2 #Dimensiones para visualización

tsne_model = TSNE(n_components=dims, random_state=42)

pca_model = PCA(n_components = 2)


embeddings_array = np.array(df_without_last_column.values.tolist())

pca_model.fit(embeddings_array)
pca_embeddings_values = pca_model.transform(embeddings_array)

tsne_embeddings_values = tsne_model.fit_transform(embeddings_array)

plt.figure(figsize=(10, 8))

clases = ['Negativos' , 'Positivos']

for i in range(len(np.unique(dataframe['class']))):
    indices = dataframe['class'] == i
    if dims == 2:
        plt.scatter( tsne_embeddings_values[indices, 0],  tsne_embeddings_values[indices, 1], label=f'{clases[i]}', alpha=0.6)

plt.xlabel('Componente t-SNE 1')
plt.ylabel('Componente t-SNE 2')
plt.title('Visualización de Embeddings mediante t-SNE')
plt.legend()
plt.show()
"""