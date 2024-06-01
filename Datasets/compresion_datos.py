import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter as cnt
import re
from operator import itemgetter

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


dataframe = pd.read_csv('Datasets/Labeled Dota 2 Player Messages Dataset.csv')


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




plt.figure(figsize=(10,6))
sns.heatmap(dataframe.isnull(), cbar=False, cmap='coolwarm')
plt.title('Mapa de Calor de Valores Nulos')

if(not dataframe.isnull().values.any()):
    print("No hay nulos")
else:
    print("hay nulos")


#plt.show()