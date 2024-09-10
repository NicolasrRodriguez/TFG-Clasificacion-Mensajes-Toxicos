from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np




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


