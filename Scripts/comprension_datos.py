import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter as cnt
import re
from sklearn.feature_extraction.text import TfidfVectorizer as tfi
import numpy as np

#from pycaret.classification import setup, compare_models  

#-----------------Funciones----------------------

def preparar_mensajes(mensaje):

    mensaje = mensaje.lower()

    mensaje = re.sub(r'[^a-z\s]', '', mensaje)

    palabras = mensaje.split()
    return palabras

def calcular_diversidad_lexica(mensaje):
    palabras = preparar_mensajes(mensaje)
    tipos = set(palabras)
    num_tokens = len(palabras)
    num_tipos = len(tipos)
    ttr = num_tipos / num_tokens
    ld = num_tipos / (num_tokens ** 0.5)
    return ttr, ld

#-----------------Carga del dataset--------------

dataframe = pd.read_csv('Data/Labeled Dota 2 Player Messages Dataset.csv')

datapack = pd.read_csv('Data/pooled_outputs.csv')

caracteristicas = datapack.drop(columns=['class'])


#--------------------Longitud--------------------
dataframe['numero_caracteres'] = dataframe['text'].apply(len)

dataframe['numero_palabras'] = dataframe['text'].apply(lambda x: len(x.split()))

longitud_media_caracteres = dataframe['numero_caracteres'].mean()
longitud_media_palabras = dataframe['numero_palabras'].mean()

mensaje_mas_largo = dataframe['numero_palabras'].max()


mensaje_mas_corto = dataframe['numero_palabras'].min()

plt.figure(figsize=(10, 6))
sns.histplot(dataframe['numero_palabras'], bins=mensaje_mas_largo, kde=False, color='darkgreen')
plt.title('Distribución de la Longitud de los Mensajes')
plt.xlabel('Longitud (palabras)')
plt.ylabel('Frecuencia')
#plt.show()

print(f'Longitud media en caracteres: {longitud_media_caracteres}')
print(f'Longitud media en palabras: {longitud_media_palabras}')

print(f'Mensasje mas largo: {mensaje_mas_largo}')
print(f'Mensasje mas corto: {mensaje_mas_corto}')


#-----------------Palabras comunes---------------
bloque_mensajes = ' '.join(dataframe['text'])

palabras = preparar_mensajes(bloque_mensajes)

contador_palabras = cnt(palabras)

palabras_mas_comunes = contador_palabras.most_common(30)


print(f'Las palabras más comunes son: {palabras_mas_comunes}')

dataframe_palabras_comunes = pd.DataFrame(palabras_mas_comunes, columns=['Palabra', 'Frecuencia'])

plt.figure(figsize=(10, 6))
sns.barplot(x='Frecuencia', y='Palabra', data=dataframe_palabras_comunes, palette='viridis')
plt.title('Palabras más comunes')
plt.xlabel('Frecuencia')
plt.ylabel('Palabra')
#plt.show()

#-----------------Diversidad lexica--------------

dataframe['TTR'], dataframe['LD'] = zip(*dataframe['text'].apply(calcular_diversidad_lexica))

print(dataframe[['cls', 'TTR', 'LD']].groupby('cls').mean())

plt.figure(figsize=(12, 6))
sns.boxplot(x='cls', y='TTR', data=dataframe)
plt.title('Índice de Tipo-Tokens (TTR) por Clase')
#plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='cls', y='LD', data=dataframe)
plt.title('Índice de Diversidad Léxica (LD) por Clase')
#plt.show()

#-----------------Palabras clave---------------

vectorizador = tfi()

tfidf_matrix = vectorizador.fit_transform(dataframe['text'])

tfidf_array = tfidf_matrix.toarray()

palabras = vectorizador.get_feature_names_out()

"""
for i, mensaje_tfidf in enumerate(tfidf_array):
    print(f"\nMensaje {i + 1} (Clase {dataframe['cls']}):")
    for palabra, tfidf in zip(palabras, mensaje_tfidf):
        if tfidf > 0:
            print(f"{palabra}: {tfidf:.4f}")
"""

df_tfidf = pd.DataFrame(tfidf_array, columns=palabras)

df_tfidf['Clase'] = dataframe['cls']

promedios_clase1 = df_tfidf[df_tfidf['Clase'] == 0].mean().drop('Clase')
promedios_clase2 = df_tfidf[df_tfidf['Clase'] == 1].mean().drop('Clase')

palabras_clave_clase1 = promedios_clase1.sort_values(ascending=False).head(20)
palabras_clave_clase2 = promedios_clase2.sort_values(ascending=False).head(20)


with open('palabras_clave.txt', 'w') as archivo:
    archivo.write("Palabras clave para en mensajes negativos:\n")
    archivo.write(str(palabras_clave_clase1))
    archivo.write("\n")
    archivo.write("Palabras clave para en mensajes positivos:\n")
    archivo.write(str(palabras_clave_clase2))

#----------------------Nulos---------------------

plt.figure(figsize=(10,6))
sns.heatmap(dataframe.isnull(), cbar=False, cmap='coolwarm')
plt.title('Mapa de Calor de Valores Nulos')

if(not dataframe.isnull().values.any()):
    print("No hay nulos")
else:
    print("hay nulos")


#-----------------Correlaciones 2 a 2-----------------

correlation_matrix = datapack.corr()


umbral = 0.9

correlation_matrix_filtered = correlation_matrix[(correlation_matrix > umbral) | (correlation_matrix < -umbral)]

mask = (correlation_matrix.abs() >= umbral) & (correlation_matrix != 1.0)



# Filtrar y mostrar las correlaciones que superen el umbral
high_correlations = correlation_matrix[mask]



f, ax = plt.subplots(figsize=(15, 12))

mask = np.triu(np.ones_like(correlation_matrix_filtered, dtype=bool))

sns.heatmap(correlation_matrix_filtered, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)

#sns.heatmap(high_correlations)


# Mostrar el mapa de calor
plt.show()
