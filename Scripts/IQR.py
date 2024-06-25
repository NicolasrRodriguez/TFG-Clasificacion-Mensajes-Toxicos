import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import pandas as pd
import numpy as np

#from official.nlp import optimization #addons da un warning

import matplotlib.pyplot as plt
import bert_model as bm

dataframe = pd.read_csv('Data/pooled_outputs.csv')

anomalias = pd.DataFrame(index=dataframe.index, columns=dataframe.columns)

#IQR para la primera característica
q1 = dataframe['0'].quantile(0.25)
q3 = dataframe['0'].quantile(0.75)
iqr = q3 - q1
limite_sup = q3 + 1.5 * iqr
limite_inf = q1 - 1.5 * iqr
print(iqr)

for column in dataframe.columns:
    anomalias[column] = (dataframe[column] < limite_inf) | (dataframe[column] > limite_sup)

anomalies_sum = anomalias.sum()
print(anomalies_sum)

# Para ver cuántas anomalías hay en cada fila
anomalies_per_row = anomalias.sum(axis=1)
print(anomalies_per_row)

# Opcional: Guardar las anomalías en un archivo CSV
anomalias.to_csv('anomalias.csv')

"""
for caracteristica in dataframe.columns:
    q1 = dataframe[caracteristica].quantile(0.25)
    q3 = dataframe[caracteristica].quantile(0.75)
    iqr = q3 - q1
    limite_sup = q3 + 1.5 * iqr
    limite_inf = q1 - 1.5 * iqr
    print(iqr)
"""