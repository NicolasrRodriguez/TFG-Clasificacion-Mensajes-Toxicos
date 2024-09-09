import tensorflow as tf
#import tensorflow_hub as hub
from sklearn.model_selection import GridSearchCV 
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.svm import SVC


np.random.seed(42)
tf.random.set_seed(42)

#----Datos---
data = pd.read_csv("Data/finaldpck.csv")
data_f = data.drop(columns=['class'])#caracteristicas
data_c = data['class']#etiquetas

#----Separación de un conjunto de prueba-------

#---Definicion del modelo---

# Modelo 1
def build_classifier_model_1(learning_rate=1e-3):
  input = tf.keras.layers.Input(shape=(169,), dtype=tf.float64, name='Entrada')#Entrada: Pooled_outputs procesadas(de 768 caracteristicas a 169)
  net = tf.keras.layers.Dropout(0.1)(input)
  net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(net)#Salida: Utiliza una función sigmoidal
  model = tf.keras.Model(input, net)
  opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)#Optimizador Adam
  loss = tf.keras.losses.BinaryCrossentropy()
  model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
  return model

model1 = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=build_classifier_model_1, verbose=0)

# Modelo 2
def build_classifier_model_2(learning_rate=1e-3):
    input = tf.keras.layers.Input(shape=(169,), dtype=tf.float64, name='Entrada')
    net = tf.keras.layers.Dense(64, activation='relu')(input)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Dropout(0.3)(net)
    net = tf.keras.layers.Dense(32, activation='relu')(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Dropout(0.3)(net)
    net = tf.keras.layers.Dense(16, activation='relu')(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Dropout(0.3)(net)
    net = tf.keras.layers.Dense(1, activation='sigmoid')(net)
    model = tf.keras.Model(input, net)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    return model

model2 = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=build_classifier_model_2, verbose=0)

# Modelo 3

svm_model = SVC()

#---Hyperparametros a probar---

# Modelos 1 y 2
param_grid_1_2 = {
    'epochs': [10, 50, 60],
    'batch_size': [ 32, 64, 128, 256],
    'learning_rate': [1e-2,1e-3, 1e-4]
}

#Modelo 3 

param_grid_3 = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
}

cv_externo = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)#CAMBIAR A 5 CUANDO ESTE TODO PRA ACABAR

cv_interno = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(estimator=svm_model , param_grid=param_grid_3, n_jobs=-1, cv=cv_interno, scoring='accuracy',verbose=1 )#GridSearch para seleccionar los hyperparametros Modelo 3
#grid = GridSearchCV(estimator=model1 , param_grid=param_grid_1_2, n_jobs=-1, cv=cv_interno, scoring='accuracy',verbose=0 )#GridSearch para seleccionar los hyperparametros Modelo 1
#grid = GridSearchCV(estimator=model2 , param_grid=param_grid_1_2, n_jobs=-1, cv=cv_interno, scoring='accuracy',verbose=0 , refit= True)#GridSearch para seleccionar los hyperparametros Modelo 2
    



#---Entrenamiento---


confusion_matrices = [] #Matrices de confusion 
best_params = [] #Mejores parametros por cada Fold externo
outer_results = [] #Resultados de cada Fold externo
accuracys = [] #Exactitudes durante el proceso de hiperparametrización
precisions = []
recalls = []
stds = [] #Desviaciones tipicas de cada iteracción
scores = []


#--Validación cruzada anidada--
for index ,(train_idx, test_idx) in enumerate(cv_externo.split(data_f, data_c)):

    X_train_outer, X_test_outer = data_f.iloc[train_idx], data_f.iloc[test_idx]
    y_train_outer, y_test_outer = data_c.iloc[train_idx], data_c.iloc[test_idx]
    
    # Realizar la búsqueda de hiperparámetros

    grid.fit(X_train_outer,y_train_outer)

    scores.append(grid.best_score_)

    stds.append(grid.cv_results_["std_test_score"][grid.best_index_])
   
    best_params.append(grid.best_params_)
    
    #Guardar los mejores parametros



   


#Mejores parametros para cada Fold
for params in best_params:
   print(params)


#Matrices de confusión de cada Fold


  
with open('resultado_cva.txt', 'w') as file:
   file.write("----------------\n")
   for index, params in enumerate(best_params):
    file.write(f"{params}\n")
    file.write(f"Puntuacion Grid Search : {scores[index]} Std: {stds[index]}\n")
    file.write("----------------\n")

#----------------------------------------------------------------------------------------------

#---Entrenamiento final----














    
