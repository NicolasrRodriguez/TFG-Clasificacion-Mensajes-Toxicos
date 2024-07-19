import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



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

atributos_redundantes = correlation(caracteristicas, umbral)

#print(len(atributos_redundantes))

caracteristicas.drop(labels= atributos_redundantes ,axis= 1,inplace=True)

#print(caracteristicas.shape)


