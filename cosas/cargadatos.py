

import tensorflow as tf
import pandas as pd

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
seed = 42

# Cargar el archivo CSV usando pandas
dataframe = pd.read_csv('Datasets/Labeled Dota 2 Player Messages Dataset.csv')

# Separar características y etiquetas
X = dataframe['text'].values
y = dataframe['cls'].values

# Obtener el tamaño del conjunto de datos
dataset_size = len(X)

# Definir el tamaño del conjunto de entrenamiento
train_size = int(0.6 * dataset_size)

# Definir el tamaño del conjunto de validación
val_size = int(0.2 * dataset_size)  # Porcentaje del conjunto de datos para validación

#Definir tamaño del conjunto de test
test_size = int(0.2 * dataset_size)

# Obtener índices aleatorios para el conjunto de validación
indices = tf.range(dataset_size)
indices = tf.random.shuffle(indices, seed=42)

# Dividir los índices en conjuntos de entrenamiento , validación y test
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size+val_size]
test_indices = indices[train_size+val_size:]


# Obtener datos de entrenamiento y validación usando los índices
X_train, y_train = X[train_indices], y[train_indices]
X_val, y_val = X[val_indices], y[val_indices]
X_test, y_test = X[test_indices], y[test_indices]

# Crear datasets TensorFlow a partir de los datos de entrenamiento y validación
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Opcional: Mezclar y dividir el conjunto de entrenamiento en lotes
train_dataset = train_dataset.shuffle(buffer_size=len(X_train), seed=42).batch(batch_size)

# Opcional: Prefetch para mejorar el rendimiento
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# No es necesario mezclar ni dividir el conjunto de validación y test
val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

# Ahora puedes usar train_dataset y val_dataset para entrenar y validar tu modelo, respectivamente
