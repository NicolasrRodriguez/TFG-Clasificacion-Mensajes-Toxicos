import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



dataframe = pd.read_csv('Data/pooled_outputs.csv')

y = dataframe['class']
dataframe_ = dataframe.drop(columns=['class'])

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(dataframe_)

plt.figure(figsize=(10, 8))

# Usar colores diferentes para cada clase
for label in np.unique(y):
    indices = y == label
    plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=f'Clase {label}', alpha=0.6)

# Añadir etiquetas y leyenda
plt.xlabel('Componente t-SNE 1')
plt.ylabel('Componente t-SNE 2')
plt.title('Visualización de Embeddings mediante t-SNE')
plt.legend()
plt.show()