import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter as cnt
import re
from sklearn.feature_extraction.text import TfidfVectorizer as tfi
import numpy as np

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

"""

frecuencias_por_clase = {}

clases = dataframe['cls'].unique()

for clase in clases:
    textos_clase = dataframe[dataframe['cls'] == clase]['text']
    todas_palabras = []
    for texto in textos_clase:
        todas_palabras.extend(preparar_mensajes(texto))
    frecuencias_por_clase[clase] = cnt(todas_palabras)


palabras_clave_por_clase = {}

for clase, frecuencias in frecuencias_por_clase.items():
    otras_clases = [c for c in clases if c != clase]
    palabras_clave_por_clase[clase] = {}
    for palabra, frecuencia in frecuencias.items():
        #otras_frecuencias = sum(frecuencias_por_clase[otra_clase][palabra] for otra_clase in otras_clases)
        if frecuencia > 5 * longitud_media_palabras:
            palabras_clave_por_clase[clase][palabra] = frecuencia

with open('palabras_clave.txt', 'w') as archivo:
    for clase, palabras_clave in palabras_clave_por_clase.items():
        archivo.write("Palabras clave para la clase " + str(clase) + ":\n")
        archivo.write(str(sorted(palabras_clave.items(), key=itemgetter(1), reverse=True)) + "\n")
        archivo.write("\n")
"""

#----------------------Nulos---------------------

plt.figure(figsize=(10,6))
sns.heatmap(dataframe.isnull(), cbar=False, cmap='coolwarm')
plt.title('Mapa de Calor de Valores Nulos')

if(not dataframe.isnull().values.any()):
    print("No hay nulos")
else:
    print("hay nulos")


#plt.show()

#-----------------Correlaciones 2 a 2-----------------

correlation_matrix = datapack.corr()

print(correlation_matrix)

f, ax = plt.subplots(figsize=(15, 12))

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, annot=True, mask=mask ,cmap="viridis")


# Mostrar el mapa de calor
plt.show()

correlations_with_target = datapack.corr()['class'].drop('class')

# Mostrar las correlaciones
print("Correlaciones con la característica respuesta (class):")
print(correlations_with_target)
