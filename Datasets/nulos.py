import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



dataframe = pd.read_csv('Datasets/Labeled Dota 2 Player Messages Dataset.csv')

plt.figure(figsize=(10,6))
sns.heatmap(dataframe.isnull(), cbar=False, cmap='coolwarm')
plt.title('Mapa de Calor de Valores Nulos')

if(not dataframe.isnull().values.any()):
    print("No hay nulos")
else:
    print("hay nulos")



plt.show()