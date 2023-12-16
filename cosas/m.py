import pandas as pd

# Lee el archivo CSV en un DataFrame de Pandas
df = pd.read_csv('tagged-data.csv')

# Selecciona las dos últimas columnas del DataFrame
df = df.iloc[:, -2:]

# Guarda el DataFrame resultante en un nuevo archivo CSV
df.to_csv('archivo_nuevo.csv', index=False)


