import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as  np



#función sacada de https://github.com/siddiquiamir/Feature-Selection/blob/main/Multicollinearity.ipynb

def correlation(df, threshold):
    correlated_cols = set()
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i]
                correlated_cols.add(colname)
    return correlated_cols



datapack = pd.read_csv('Data/pooled_outputs.csv')

caracteristicas = datapack.drop(columns=['class'])

correlation_matrix = caracteristicas.corr()

umbral = 0.9

high_corr_pos = (correlation_matrix> umbral) & (correlation_matrix != 1)

print(high_corr_pos)

high_corr_neg = (correlation_matrix< -umbral) & (correlation_matrix != 1)

# Contar el número de correlaciones altas (donde el valor es True)
num_high_corr_pos = high_corr_pos.sum().sum() // 2  # Dividimos entre 2 para no contar correlaciones duplicadas

# Contar el número de correlaciones altas (donde el valor es True)
num_high_corr_neg = high_corr_neg.sum().sum() // 2  # Dividimos entre 2 para no contar correlaciones duplicadas

print(f"El número de atributos con una correlación mayor a 0.9 es: {num_high_corr_pos} y menor a 0.9 es {num_high_corr_neg}")

atributos_redundantes = correlation(caracteristicas, umbral)

atributos_redundantes_array = np.array(list(atributos_redundantes))

print(len(atributos_redundantes_array))

print(atributos_redundantes_array)

#print(len(atributos_redundantes))

caracteristicas.drop(labels= atributos_redundantes ,axis= 1,inplace=True)

#print(caracteristicas.shape)


